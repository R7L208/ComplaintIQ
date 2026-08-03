/*
 *  csv_to_parquet.cpp
 *
 *  Command line ETL tool that reads a CFPB complaint CSV file, transforms it
 *  into the ML-ready schema defined in complaintiq/schema.h (renamed columns,
 *  a normalized narrative, and derived features/label), optionally drops rows
 *  that have no consumer narrative, and writes the result out as a Parquet file
 *  using Apache Arrow and Parquet.
 */

// Project headers: the shared schema contract defines the output columns and the
// raw source column names this ETL maps onto them.
#include "complaintiq/schema.h"

// Third-party libraries: Apache Arrow and Parquet provide the columnar table
// format, CSV reader, compute kernels, and Parquet writer used by this ETL.
#include <arrow/api.h>             // for core Arrow types like Table and arrays
#include <arrow/compute/api.h>     // for the Filter compute function
#include <arrow/csv/api.h>         // for reading CSV files into Arrow tables
#include <arrow/io/api.h>          // for Arrow file input and output streams
#include <parquet/arrow/writer.h>  // for writing Arrow tables to Parquet
#include <parquet/properties.h>    // for Parquet writer properties

#include <filesystem>     // for creating directories and handling paths
#include <iostream>       // for writing status messages to std::cerr
#include <memory>         // for std::shared_ptr
#include <string>         // for std::string
#include <unordered_map>  // for mapping output column names to their arrays
#include <vector>         // for std::vector of Arrow chunks and columns

#include <cctype>   // for std::isspace and std::tolower
#include <cstddef>  // for std::size_t
#include <cstdint>  // for int32_t and int8_t
#include <cstdio>   // for std::sscanf when reading the date fields
#include <cstdlib>  // for std::exit
#include <ctime>    // for std::tm and timegm when converting dates to day counts

struct CliOptions {
  std::string inputPath;
  std::string outputPath;
  bool requireNarrative = false;
};

// The raw CFPB CSV header for the free-text company response. It is a passthrough
// column and also the source the monetary_relief label is derived from.
const std::string COMPANY_RESPONSE_SOURCE = "Company response to consumer";

// The raw CFPB CSV header for the Yes/No flag that becomes a boolean column.
const std::string TIMELY_RESPONSE_SOURCE = "Timely response?";

// The raw CFPB CSV date headers. CFPB now exports these as ISO-8601 timestamps
// (e.g. "2025-01-26T00:45:30.000Z"), so they are read as strings and parsed into
// date32 rather than passed through Arrow's date reader (which rejects the time
// component).
const std::string DATE_RECEIVED_SOURCE = "Date received";
const std::string DATE_SENT_TO_COMPANY_SOURCE = "Date sent to company";

// A single raw-to-output column mapping. Each passthrough column is read as its
// target Arrow type and copied straight across under its snake_case output name,
// so numeric-looking fields become real numbers and dates become real dates
// while free-text fields stay strings.
struct ColumnSpec {
  std::string sourceName;
  std::string outputName;
  std::shared_ptr<arrow::DataType> readType;
};

// The passthrough columns are copied across unchanged (under their output names
// and target types). The consumer narrative, the date columns, the Yes/No flag,
// and the label are handled separately because they are transformed rather than
// copied.
const std::vector<ColumnSpec>& PassthroughColumns() {
  static const std::vector<ColumnSpec> columns = {
    {"Complaint ID", complaintiq::columns::COMPLAINT_ID, arrow::int64()},
    {"Product", complaintiq::columns::PRODUCT, arrow::utf8()},
    {"Sub-product", complaintiq::columns::SUB_PRODUCT, arrow::utf8()},
    {"Issue", complaintiq::columns::ISSUE, arrow::utf8()},
    {"Sub-issue", complaintiq::columns::SUB_ISSUE, arrow::utf8()},
    {"Company", complaintiq::columns::COMPANY, arrow::utf8()},
    {"State", complaintiq::columns::STATE, arrow::utf8()},
    {"ZIP code", complaintiq::columns::ZIP_CODE, arrow::utf8()},
    {"Tags", complaintiq::columns::TAGS, arrow::utf8()},
    {"Submitted via", complaintiq::columns::SUBMITTED_VIA, arrow::utf8()},
    {COMPANY_RESPONSE_SOURCE, complaintiq::columns::COMPANY_RESPONSE_TO_CONSUMER, arrow::utf8()}
  };
  return columns;
}

void PrintUsage( const char* pProgramName ) {
  std::cerr
      << "Usage: " << pProgramName
      << "--input <input_csv_path"
      << "--output <output_parquet_path"
      << "[--require-narrative]\n";
}

CliOptions ParseArgs( int argc, char** argv ) {
  CliOptions options;

  // Command line arguments are parsed manually.
  for( int i = 1; i < argc; i++ ) {
    const std::string arg = argv[i];

    if( arg == "--input" && i + 1 < argc ) {
      options.inputPath = argv[++i];
    } else if( arg == "--output" && i + 1 < argc ) {
      options.outputPath = argv[++i];
    } else if( arg == "--require-narrative" ) {
      options.requireNarrative = true;
    } else if( arg == "-h" || arg == "--help" ) {
      PrintUsage( argv[0] );
      std::exit( 0 );
    } else {
      std::cerr << "Unkonwn or incomplete argument: " << arg << std::endl;
      PrintUsage( argv[0] );
      std::exit( 1 );
    }
  }

  // The ETL cannot safely run without both paths because Arrow needs a concrete
  // input file to read from and Parquet needs a concrete output file to write to.
  if( options.inputPath.empty() || options.outputPath.empty() ) {
    std::cerr << "Both --input and --output are required" << std::endl;
    PrintUsage( argv[0] );
    std::exit( 1 );
  }

  return options;
}

std::string Trim( const std::string& value ) {
  // Trimming whitespace lets the narrative filter treat strings containing only
  // spaces, tabs, or newlines the same as empty narratives.
  std::size_t start = 0;
  while( start < value.size() && std::isspace( static_cast<unsigned char>( value[start] ) ) ) {
    ++start;
  }

  std::size_t end = value.size();
  while( end > start && std::isspace( static_cast<unsigned char>( value[end - 1] ) ) ) {
    --end;
  }

  return value.substr( start, end - start );
}

std::string ToLower( const std::string& value ) {
  std::string lowered = value;
  for( char& character : lowered ) {
    character = static_cast<char>( std::tolower( static_cast<unsigned char>( character ) ) );
  }

  return lowered;
}

// CFPB redacts personal information with runs of the capital letter X (for
// example "XXXX" or "XX/XX/XXXX"). Collapsing each redaction run into a single
// stable token keeps the redactions from dominating text-model vocabularies.
std::string NormalizeNarrative( const std::string& value ) {
  const std::string redactionToken = "redacted_token";

  std::string redacted;
  redacted.reserve( value.size() );

  std::size_t i = 0;
  while( i < value.size() ) {
    if( value[i] == 'X' ) {
      std::size_t runEnd = i;
      while( runEnd < value.size() && value[runEnd] == 'X' ) {
        ++runEnd;
      }

      // Only runs of two or more X's are treated as redactions so a lone X in
      // ordinary text is left alone.
      if( runEnd - i >= 2 ) {
        redacted += redactionToken;
      } else {
        redacted.append( value, i, runEnd - i );
      }

      i = runEnd;
    } else {
      redacted += value[i];
      ++i;
    }
  }

  // Lowercasing standardizes the text so downstream TF-IDF style features do not
  // treat differently cased words as distinct tokens.
  return ToLower( redacted );
}

// monetary_relief is the ML target label. It is 1 only when the company response
// granted monetary relief, and 0 otherwise. "Closed with non-monetary relief"
// must map to 0 even though it contains the substring "monetary relief".
int8_t MonetaryReliefLabel( const std::string& response ) {
  const std::string lowered = ToLower( response );

  const bool hasMonetaryRelief = lowered.find( "monetary relief" ) != std::string::npos;
  const bool hasNonMonetary = lowered.find( "non-monetary" ) != std::string::npos;

  return ( hasMonetaryRelief && !hasNonMonetary ) ? 1 : 0;
}

// https://arrow.apache.org/docs/cpp/api/formats.html
arrow::Result<std::shared_ptr<arrow::Table>> ReadCsvAsTable(
    const std::string& inputPath
) {
  // Arrow uses Result<T> for operations that can fail while still returning a
  // value. This macro either assigns the value or immediately returns the error.
  // https://arrow.apache.org/docs/cpp/api/support.html#result_8h_1aaf9efe9debc83022fad249a0fa56e680
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::io::ReadableFile> inputFile,
      arrow::io::ReadableFile::Open( inputPath )
  );

  // ReadOptions control file-level behavior. Threads are enabled so Arrow can
  // read and convert larger CSV files more efficiently when possible.
  arrow::csv::ReadOptions readOptions = arrow::csv::ReadOptions::Defaults();
  readOptions.use_threads = true;

  // CFPB complaint narratives can contain embedded newlines, so the parser must
  // allow newline characters inside quoted CSV values instead of treating every
  // newline as the end of a row.
  arrow::csv::ParseOptions parseOptions = arrow::csv::ParseOptions::Defaults();
  parseOptions.newlines_in_values = true;

  // ConvertOptions control how raw CSV text becomes typed Arrow arrays. These
  // settings allow empty string-like values to become nulls during conversion.
  arrow::csv::ConvertOptions convertOptions = arrow::csv::ConvertOptions::Defaults();
  convertOptions.strings_can_be_null = true;
  convertOptions.quoted_strings_can_be_null = true;

  // Read each passthrough column as its target type so complaint_id becomes an
  // integer and the dates become real dates, while the ZIP code and other codes
  // stay strings to preserve leading zeros. The narrative and Yes/No flags are
  // read as strings here because they are transformed further below.
  for( const ColumnSpec& column : PassthroughColumns() ) {
    convertOptions.column_types[column.sourceName] = column.readType;
  }
  convertOptions.column_types[complaintiq::raw::SOURCE_NARRATIVE_COLUMN] = arrow::utf8();
  convertOptions.column_types[TIMELY_RESPONSE_SOURCE] = arrow::utf8();

  // The date columns are read as strings so the ISO-8601 timestamps CFPB now
  // exports can be parsed into date32 below instead of failing Arrow's stricter
  // date reader.
  convertOptions.column_types[DATE_RECEIVED_SOURCE] = arrow::utf8();
  convertOptions.column_types[DATE_SENT_TO_COMPANY_SOURCE] = arrow::utf8();

  // TableReader converts the CSV into an Arrow Table, which is a columnar,
  // in-memory format. This matches Parquet's columnar layout and avoids building
  // a custom row-by-row representation.
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::csv::TableReader> csvReader,
      arrow::csv::TableReader::Make(
          arrow::io::default_io_context(),
          inputFile,
          readOptions,
          parseOptions,
          convertOptions
      )
  );

  return csvReader->Read();
}

// Locating columns by name keeps the ETL working even if the CFPB export changes
// column order, and gives a clear error when an expected column is missing.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> GetColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& columnName
) {
  std::shared_ptr<arrow::ChunkedArray> column = table->GetColumnByName( columnName );

  if( !column ) {
    return arrow::Status::Invalid( "Missing expected column: ", columnName );
  }

  return column;
}

// The narrative-derived columns all depend on the same per-row inspection of the
// consumer narrative, so they are built together in a single pass. complaint_text
// holds the normalized narrative (null when blank), has_narrative records whether
// usable text remained after trimming, and complaint_text_length is that text's
// length (0 for blank narratives).
struct NarrativeColumns {
  std::shared_ptr<arrow::ChunkedArray> complaintText;
  std::shared_ptr<arrow::ChunkedArray> complaintTextLength;
  std::shared_ptr<arrow::ChunkedArray> hasNarrative;
};

arrow::Result<NarrativeColumns> BuildNarrativeColumns(
    const std::shared_ptr<arrow::Table>& table
) {
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> narrativeColumn,
      GetColumn( table, complaintiq::raw::SOURCE_NARRATIVE_COLUMN )
  );

  // Each output column is built as one array per input chunk and gathered into a
  // chunked array at the end. Building complaint_text this way keeps any single
  // array under the ~2 GB limit a utf8 array has (its offsets are 32-bit), which
  // the full dataset's narratives would otherwise blow past. The chunk boundaries
  // just follow the CSV reader's, so row order is preserved without a costly
  // combine step.
  arrow::ArrayVector complaintTextChunks;
  arrow::ArrayVector complaintTextLengthChunks;
  arrow::ArrayVector hasNarrativeChunks;

  for( const std::shared_ptr<arrow::Array>& chunk : narrativeColumn->chunks() ) {
    // The narrative transform depends on string operations, so the column type
    // must be checked before casting the generic Arrow Array to a StringArray.
    if( chunk->type_id() != arrow::Type::STRING ) {
      return arrow::Status::Invalid(
          "Expected narrative column to be string but got: ",
          chunk->type()->ToString()
      );
    }

    std::shared_ptr<arrow::StringArray> stringArray =
        std::static_pointer_cast<arrow::StringArray>( chunk );

    // Fresh builders for each chunk, so a finished array only ever holds one
    // chunk's worth of rows instead of growing across the whole dataset.
    arrow::StringBuilder complaintTextBuilder;
    arrow::Int32Builder complaintTextLengthBuilder;
    arrow::BooleanBuilder hasNarrativeBuilder;

    for( int64_t i = 0; i < stringArray->length(); i++ ) {
      // A null or whitespace-only narrative carries no usable text for modeling,
      // so complaint_text becomes null, the length is 0, and has_narrative false.
      std::string trimmed;
      if( !stringArray->IsNull( i ) ) {
        trimmed = Trim( stringArray->GetString( i ) );
      }

      if( trimmed.empty() ) {
        ARROW_RETURN_NOT_OK( complaintTextBuilder.AppendNull() );
        ARROW_RETURN_NOT_OK( complaintTextLengthBuilder.Append( 0 ) );
        ARROW_RETURN_NOT_OK( hasNarrativeBuilder.Append( false ) );
      } else {
        const std::string normalized = NormalizeNarrative( trimmed );
        ARROW_RETURN_NOT_OK( complaintTextBuilder.Append( normalized ) );
        ARROW_RETURN_NOT_OK(
            complaintTextLengthBuilder.Append( static_cast<int32_t>( normalized.size() ) )
        );
        ARROW_RETURN_NOT_OK( hasNarrativeBuilder.Append( true ) );
      }
    }

    // Finish this chunk's arrays and collect them; the next chunk starts over.
    std::shared_ptr<arrow::Array> complaintTextArray;
    ARROW_RETURN_NOT_OK( complaintTextBuilder.Finish( &complaintTextArray ) );
    complaintTextChunks.push_back( complaintTextArray );

    std::shared_ptr<arrow::Array> complaintTextLengthArray;
    ARROW_RETURN_NOT_OK( complaintTextLengthBuilder.Finish( &complaintTextLengthArray ) );
    complaintTextLengthChunks.push_back( complaintTextLengthArray );

    std::shared_ptr<arrow::Array> hasNarrativeArray;
    ARROW_RETURN_NOT_OK( hasNarrativeBuilder.Finish( &hasNarrativeArray ) );
    hasNarrativeChunks.push_back( hasNarrativeArray );
  }

  // Assemble each column from its per-chunk arrays. Passing the types explicitly
  // keeps the columns well-formed even if the input happened to have no chunks.
  NarrativeColumns result;
  result.complaintText =
      std::make_shared<arrow::ChunkedArray>( complaintTextChunks, arrow::utf8() );
  result.complaintTextLength =
      std::make_shared<arrow::ChunkedArray>( complaintTextLengthChunks, arrow::int32() );
  result.hasNarrative =
      std::make_shared<arrow::ChunkedArray>( hasNarrativeChunks, arrow::boolean() );

  return result;
}

// monetary_relief is derived from the free-text company response, so it is built
// with a per-row pass just like the narrative columns.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> BuildMonetaryReliefColumn(
    const std::shared_ptr<arrow::Table>& table
) {
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> responseColumn,
      GetColumn( table, COMPANY_RESPONSE_SOURCE )
  );

  arrow::Int8Builder monetaryReliefBuilder;

  for( const std::shared_ptr<arrow::Array>& chunk : responseColumn->chunks() ) {
    if( chunk->type_id() != arrow::Type::STRING ) {
      return arrow::Status::Invalid(
          "Expected company response column to be string but got: ",
          chunk->type()->ToString()
      );
    }

    std::shared_ptr<arrow::StringArray> stringArray =
        std::static_pointer_cast<arrow::StringArray>( chunk );

    for( int64_t i = 0; i < stringArray->length(); i++ ) {
      // A missing response cannot indicate monetary relief, so it defaults to 0.
      const int8_t label = stringArray->IsNull( i )
          ? 0
          : MonetaryReliefLabel( stringArray->GetString( i ) );
      ARROW_RETURN_NOT_OK( monetaryReliefBuilder.Append( label ) );
    }
  }

  std::shared_ptr<arrow::Array> monetaryReliefArray;
  ARROW_RETURN_NOT_OK( monetaryReliefBuilder.Finish( &monetaryReliefArray ) );

  return std::make_shared<arrow::ChunkedArray>( monetaryReliefArray );
}

// CFPB dates come as ISO-8601 strings like "2025-01-26T00:45:30.000Z". date32
// stores a plain calendar date with no time, so this reads the YYYY-MM-DD part
// from the front of each string and converts it to the day count date32 uses.
// timegm gives the seconds since 1970-01-01 for a UTC date, so dividing by the
// number of seconds in a day yields that day count. A blank or unparseable date
// is stored as null instead of a wrong day.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> BuildDateColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& sourceColumnName
) {
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> dateColumn,
      GetColumn( table, sourceColumnName )
  );

  arrow::Date32Builder dateBuilder;
  const int64_t secondsPerDay = 24 * 60 * 60;

  for( const std::shared_ptr<arrow::Array>& chunk : dateColumn->chunks() ) {
    if( chunk->type_id() != arrow::Type::STRING ) {
      return arrow::Status::Invalid(
          "Expected date column '",
          sourceColumnName,
          "' to be string but got: ",
          chunk->type()->ToString()
      );
    }

    std::shared_ptr<arrow::StringArray> stringArray =
        std::static_pointer_cast<arrow::StringArray>( chunk );

    for( int64_t i = 0; i < stringArray->length(); i++ ) {
      int year = 0;
      int month = 0;
      int day = 0;

      // sscanf reads just the three date fields off the front of the string and
      // ignores the time component. A null or unmatched value becomes a null date.
      if( stringArray->IsNull( i ) ||
          std::sscanf( stringArray->GetString( i ).c_str(), "%4d-%2d-%2d", &year, &month, &day ) != 3 ) {
        ARROW_RETURN_NOT_OK( dateBuilder.AppendNull() );
        continue;
      }

      std::tm calendarDate = {};
      calendarDate.tm_year = year - 1900;  // std::tm counts years from 1900.
      calendarDate.tm_mon = month - 1;     // std::tm months are 0-based.
      calendarDate.tm_mday = day;

      const int32_t days = static_cast<int32_t>( timegm( &calendarDate ) / secondsPerDay );
      ARROW_RETURN_NOT_OK( dateBuilder.Append( days ) );
    }
  }

  std::shared_ptr<arrow::Array> dateArray;
  ARROW_RETURN_NOT_OK( dateBuilder.Finish( &dateArray ) );

  return std::make_shared<arrow::ChunkedArray>( dateArray );
}

// The CFPB Yes/No flags are stored as free text, so Arrow's CSV reader cannot
// parse them as booleans directly. "Yes" becomes true and "No" becomes false;
// anything else (including "N/A", blanks, and nulls) becomes null so downstream
// code can distinguish an unknown flag from a real false.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> BuildYesNoBooleanColumn(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& sourceColumnName
) {
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> sourceColumn,
      GetColumn( table, sourceColumnName )
  );

  arrow::BooleanBuilder booleanBuilder;

  for( const std::shared_ptr<arrow::Array>& chunk : sourceColumn->chunks() ) {
    if( chunk->type_id() != arrow::Type::STRING ) {
      return arrow::Status::Invalid(
          "Expected column '",
          sourceColumnName,
          "' to be string but got: ",
          chunk->type()->ToString()
      );
    }

    std::shared_ptr<arrow::StringArray> stringArray =
        std::static_pointer_cast<arrow::StringArray>( chunk );

    for( int64_t i = 0; i < stringArray->length(); i++ ) {
      if( stringArray->IsNull( i ) ) {
        ARROW_RETURN_NOT_OK( booleanBuilder.AppendNull() );
        continue;
      }

      const std::string value = ToLower( Trim( stringArray->GetString( i ) ) );
      if( value == "yes" ) {
        ARROW_RETURN_NOT_OK( booleanBuilder.Append( true ) );
      } else if( value == "no" ) {
        ARROW_RETURN_NOT_OK( booleanBuilder.Append( false ) );
      } else {
        ARROW_RETURN_NOT_OK( booleanBuilder.AppendNull() );
      }
    }
  }

  std::shared_ptr<arrow::Array> booleanArray;
  ARROW_RETURN_NOT_OK( booleanBuilder.Finish( &booleanArray ) );

  return std::make_shared<arrow::ChunkedArray>( booleanArray );
}

// Assemble the ML-ready table by renaming the passthrough columns, plugging in
// the derived columns, and emitting them in exactly the order and types declared
// in the shared schema contract.
arrow::Result<std::shared_ptr<arrow::Table>> TransformToContract(
    const std::shared_ptr<arrow::Table>& table
) {
  std::unordered_map<std::string, std::shared_ptr<arrow::ChunkedArray>> outputColumns;

  for( const ColumnSpec& column : PassthroughColumns() ) {
    ARROW_ASSIGN_OR_RAISE(
        std::shared_ptr<arrow::ChunkedArray> sourceColumn,
        GetColumn( table, column.sourceName )
    );
    outputColumns[column.outputName] = sourceColumn;
  }

  ARROW_ASSIGN_OR_RAISE( NarrativeColumns narrative, BuildNarrativeColumns( table ) );
  outputColumns[complaintiq::columns::COMPLAINT_TEXT] = narrative.complaintText;
  outputColumns[complaintiq::columns::COMPLAINT_TEXT_LENGTH] = narrative.complaintTextLength;
  outputColumns[complaintiq::columns::HAS_NARRATIVE] = narrative.hasNarrative;

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> dateReceived,
      BuildDateColumn( table, DATE_RECEIVED_SOURCE )
  );
  outputColumns[complaintiq::columns::DATE_RECEIVED] = dateReceived;

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> dateSentToCompany,
      BuildDateColumn( table, DATE_SENT_TO_COMPANY_SOURCE )
  );
  outputColumns[complaintiq::columns::DATE_SENT_TO_COMPANY] = dateSentToCompany;

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> timelyResponse,
      BuildYesNoBooleanColumn( table, TIMELY_RESPONSE_SOURCE )
  );
  outputColumns[complaintiq::columns::TIMELY_RESPONSE] = timelyResponse;

  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> monetaryRelief,
      BuildMonetaryReliefColumn( table )
  );
  outputColumns[complaintiq::columns::MONETARY_RELIEF] = monetaryRelief;

  // Driving the field/column order from getExpectedSchema() guarantees the ETL
  // output stays aligned with the contract the validators check.
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

  for( const complaintiq::ExpectedColumn& expected : complaintiq::getExpectedSchema() ) {
    fields.push_back( arrow::field( expected.name, expected.type ) );
    columns.push_back( outputColumns.at( expected.name ) );
  }

  std::shared_ptr<arrow::Schema> schema = arrow::schema( fields );

  return arrow::Table::Make( schema, columns, table->num_rows() );
}

arrow::Result<std::shared_ptr<arrow::Table>> FilterNarrativeRows(
    const std::shared_ptr<arrow::Table>& table
) {
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::ChunkedArray> mask,
      GetColumn( table, complaintiq::columns::HAS_NARRATIVE )
  );

  // arrow::compute::Filter applies the boolean has_narrative column as a mask in
  // Arrow's execution engine, keeping the transformation columnar instead of
  // manually copying rows.
  ARROW_ASSIGN_OR_RAISE(
      arrow::Datum filtered,
      arrow::compute::Filter( arrow::Datum( table ), arrow::Datum( mask ) )
  );

  return filtered.table();
}

arrow::Status WriteParquetTable(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& outputPath
) {
  const std::filesystem::path outputFsPath( outputPath );
  const std::filesystem::path outputParent = outputFsPath.parent_path();

  // Creating the parent directory makes the ETL easier to run from scripts
  // because callers do not need to prepare the output folder beforehand.
  if( !outputParent.empty() ) {
    std::filesystem::create_directories( outputParent );
  }

  // FileOutputStream gives the Parquet writer an Arrow-compatible output target.
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::io::FileOutputStream> outputFile,
      arrow::io::FileOutputStream::Open( outputPath )
  );

  // Parquet stores data in row groups. A moderate row group size keeps the file
  // readable in chunks while still allowing columnar compression and scans.
  const int64_t rowGroupSize = 64 * 1024;

  // WriteTable converts the in-memory Arrow Table into an on-disk Parquet file.
  // The default memory pool lets Arrow manage temporary buffers consistently.
  ARROW_RETURN_NOT_OK(
      parquet::arrow::WriteTable(
          *table,
          arrow::default_memory_pool(),
          outputFile,
          rowGroupSize
      )
  );

  // Closing the stream flushes buffered bytes so the Parquet file is complete on
  // disk before the program reports success.
  ARROW_RETURN_NOT_OK( outputFile->Close() );

  return arrow::Status::OK();
}

arrow::Status ConvertCsvToParquet( const CliOptions& options ) {
  // Checking the input path early gives a clearer error than letting the lower
  // level CSV reader fail later.
  if( !std::filesystem::exists( options.inputPath ) ) {
    return arrow::Status::IOError( "Input CSV does not exist...", options.inputPath );
  }

  // The CSV is loaded into an Arrow Table first so the transform, optional
  // filtering, and Parquet writing can all operate on the same columnar data.
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::Table> rawTable,
      ReadCsvAsTable( options.inputPath )
  );

  const int64_t originalRows = rawTable->num_rows();

  // Transforming the raw CFPB columns into the ML-ready contract is always done
  // so the Parquet output is consistent regardless of the filtering option.
  ARROW_ASSIGN_OR_RAISE(
      std::shared_ptr<arrow::Table> table,
      TransformToContract( rawTable )
  );

  // Filtering is optional so the same executable can either preserve the full
  // complaint dataset or keep only rows useful for narrative-based modeling.
  if( options.requireNarrative ) {
    ARROW_ASSIGN_OR_RAISE( table, FilterNarrativeRows( table ) );
  }

  ARROW_RETURN_NOT_OK( WriteParquetTable( table, options.outputPath ) );

  // Reporting both input and output row counts makes it easy to verify how many
  // records were removed when the narrative filter is enabled.
  std::cerr
      << "Converted CSV to Parquet" << std::endl
      << "Input:         " << options.inputPath << std::endl
      << "Output:        " << options.outputPath << std::endl
      << "Input Rows:    " << originalRows << std::endl
      << "Output Rows:   " << table->num_rows() << std::endl
      << "Columns:       " << table->num_columns() << std::endl;

  return arrow::Status::OK();
}

int main(int argc, char* argv[]) {
  const CliOptions options = ParseArgs( argc, argv );

  // Keeping the ETL work outside main makes main responsible only for program
  // flow: parse arguments, run conversion, and translate success/failure into an
  // operating system exit code.
  const arrow::Status status = ConvertCsvToParquet( options );

  if( !status.ok() ) {
    std::cerr << "ETL failed: " << status.ToString() << std::endl;
    return 1;
  }

  return 0;  // Report success to the operating system.
}
