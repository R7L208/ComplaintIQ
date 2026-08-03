/*
 *  validate_parquet_values.cpp
 *
 *  Small validator executable that opens the Parquet file produced by the ETL
 *  and checks that individual column values match the expected results for the
 *  fixed test fixture, covering the derived ML label and engineered features.
 */

// Project headers: the shared schema contract defines the output column names.
#include "complaintiq/schema.h"  // for the complaintiq::columns name constants

// Third-party libraries: Apache Arrow and Parquet provide the columnar table
// format and the Parquet reader used to inspect the ETL output.
#include <arrow/api.h>              // for core Arrow types like Table and arrays
#include <arrow/io/api.h>           // for Arrow file input streams
#include <parquet/arrow/reader.h>   // for reading Parquet files into Arrow

#include <iostream>  // for std::cerr and std::cout messages
#include <memory>    // for std::shared_ptr and std::unique_ptr
#include <string>    // for std::string
#include <vector>    // for std::vector of expected values

#include <cstdint>  // for int64_t and int8_t

// These expected values match the three-row fixture created by
// test_cpp_etl_schema.sh. Keeping them as constants makes it clear this
// validator is testing deterministic ETL behavior, not the live CFPB dataset.
const int EXPECTED_ROW_COUNT = 3;
const int FIRST_ROW_INDEX = 0;
const int SECOND_ROW_INDEX = 1;
const int THIRD_ROW_INDEX = 2;

// Use Arrow and Parquet directly so this test does not introduce Python,
// DuckDB, or other dependencies outside the C++ ETL stack.
arrow::Result<std::shared_ptr<arrow::Table>> readParquetTable( const std::string& parquetPath ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::io::ReadableFile> inputFile,
    arrow::io::ReadableFile::Open( parquetPath )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::unique_ptr<parquet::arrow::FileReader> parquetReader,
    parquet::arrow::OpenFile( inputFile, arrow::default_memory_pool() )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Table> table,
    parquetReader->ReadTable()
  );

  // Combine chunks so row-based validation can read from one array per column.
  // The test fixture is small, and this keeps the validator easier to read.
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Table> combinedTable,
    table->CombineChunks( arrow::default_memory_pool() )
  );

  return combinedTable;
}

void printUsage( const char* pProgramName ) {
  std::cerr << "Usage: " << pProgramName << " <parquet_file_path>" << std::endl;
}

// Fail early when a required column is missing so later value checks can assume
// the expected ML-ready schema is present.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> getRequiredColumn(
  const std::shared_ptr<arrow::Table>& table,
  const std::string& columnName
) {
  std::shared_ptr<arrow::ChunkedArray> column = table->GetColumnByName( columnName );

  if( column == nullptr ) {
    return arrow::Status::Invalid( "Missing expected column: ", columnName );
  }

  return column;
}

// The validators expect CombineChunks() to leave each column with a single
// chunk. This keeps row indexing simple and avoids adding a separate row
// traversal helper for this small fixture.
arrow::Result<std::shared_ptr<arrow::Array>> getOnlyChunk(
  const std::shared_ptr<arrow::ChunkedArray>& column,
  const std::string& columnName
) {
  if( column->num_chunks() != 1 ) {
    return arrow::Status::Invalid(
      "Expected one chunk for column '",
      columnName,
      "', got ",
      column->num_chunks()
    );
  }

  return column->chunk( 0 );
}

arrow::Result<std::shared_ptr<arrow::StringArray>> getStringColumn(
  const std::shared_ptr<arrow::Table>& table,
  const std::string& columnName
) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::ChunkedArray> column,
    getRequiredColumn( table, columnName )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Array> array,
    getOnlyChunk( column, columnName )
  );

  if( array->type_id() != arrow::Type::STRING ) {
    return arrow::Status::Invalid(
      "Column '",
      columnName,
      "' should be string but got ",
      array->type()->ToString()
    );
  }

  return static_pointer_cast<arrow::StringArray>( array );
}

arrow::Result<std::shared_ptr<arrow::BooleanArray>> getBooleanColumn(
  const std::shared_ptr<arrow::Table>& table,
  const std::string& columnName
) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::ChunkedArray> column,
    getRequiredColumn( table, columnName )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Array> array,
    getOnlyChunk( column, columnName )
  );

  if( array->type_id() != arrow::Type::BOOL ) {
    return arrow::Status::Invalid(
      "Column '",
      columnName,
      "' should be boolean but got ",
      array->type()->ToString()
    );
  }

  return static_pointer_cast<arrow::BooleanArray>( array );
}

arrow::Result<std::shared_ptr<arrow::Int8Array>> getInt8Column(
  const std::shared_ptr<arrow::Table>& table,
  const std::string& columnName
) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::ChunkedArray> column,
    getRequiredColumn( table, columnName )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Array> array,
    getOnlyChunk( column, columnName )
  );

  if( array->type_id() != arrow::Type::INT8 ) {
    return arrow::Status::Invalid(
      "Column '",
      columnName,
      "' should be int8 but got ",
      array->type()->ToString()
    );
  }

  return static_pointer_cast<arrow::Int8Array>( array );
}

arrow::Result<std::shared_ptr<arrow::Int32Array>> getInt32Column(
  const std::shared_ptr<arrow::Table>& table,
  const std::string& columnName
) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::ChunkedArray> column,
    getRequiredColumn( table, columnName )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Array> array,
    getOnlyChunk( column, columnName )
  );

  if( array->type_id() != arrow::Type::INT32 ) {
    return arrow::Status::Invalid(
      "Column '",
      columnName,
      "' should be int32 but got ",
      array->type()->ToString()
    );
  }

  return static_pointer_cast<arrow::Int32Array>( array );
}

arrow::Result<std::shared_ptr<arrow::Date32Array>> getDate32Column(
  const std::shared_ptr<arrow::Table>& table,
  const std::string& columnName
) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::ChunkedArray> column,
    getRequiredColumn( table, columnName )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Array> array,
    getOnlyChunk( column, columnName )
  );

  if( array->type_id() != arrow::Type::DATE32 ) {
    return arrow::Status::Invalid(
      "Column '",
      columnName,
      "' should be date32 but got ",
      array->type()->ToString()
    );
  }

  return static_pointer_cast<arrow::Date32Array>( array );
}

// The fixture has three rows, and the value checks below are written against
// those exact rows. This guards against accidentally validating a partial file.
arrow::Status validateRowCount( const std::shared_ptr<arrow::Table>& table ) {
  const int64_t actualRowCount = table->num_rows();

  if( actualRowCount != EXPECTED_ROW_COUNT ) {
    return arrow::Status::Invalid(
      "Expected ",
      EXPECTED_ROW_COUNT,
      " rows, got ",
      actualRowCount
    );
  }

  return arrow::Status::OK();
}

// monetary_relief is the target label for the ML project, so this test verifies
// that it is derived from company_response_to_consumer correctly.
arrow::Status validateMonetaryRelief( const std::shared_ptr<arrow::Table>& table ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Int8Array> monetaryRelief,
    getInt8Column( table, complaintiq::columns::MONETARY_RELIEF )
  );

  const std::vector<int8_t> expectedValues = {1, 0, 0};

  for( int rowIndex = 0; rowIndex < EXPECTED_ROW_COUNT; rowIndex++ ) {
    const int8_t actualValue = monetaryRelief->Value( rowIndex );
    const int8_t expectedValue = expectedValues.at( rowIndex );

    if( actualValue != expectedValue ) {
      return arrow::Status::Invalid(
        "monetary_relief row ",
        rowIndex,
        " mismatch. Expected ",
        static_cast<int>( expectedValue ),
        ", got ",
        static_cast<int>( actualValue )
      );
    }
  }

  return arrow::Status::OK();
}

// has_narrative should reflect usable text after trimming, not merely whether
// the raw CSV field exists.
arrow::Status validateHasNarrative( const std::shared_ptr<arrow::Table>& table ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::BooleanArray> hasNarrative,
    getBooleanColumn( table, complaintiq::columns::HAS_NARRATIVE )
  );

  const std::vector<bool> expectedValues = {true, false, true};

  for( int rowIndex = 0; rowIndex < EXPECTED_ROW_COUNT; rowIndex++ ) {
    const bool actualValue = hasNarrative->Value( rowIndex );
    const bool expectedValue = expectedValues.at( rowIndex );

    if( actualValue != expectedValue ) {
      return arrow::Status::Invalid(
        "has_narrative row ",
        rowIndex,
        " mismatch."
      );
    }
  }

  return arrow::Status::OK();
}

// complaint_text should be normalized for text modeling. Blank narratives
// should become null so downstream TF-IDF logic can treat them consistently.
arrow::Status validateComplaintText( const std::shared_ptr<arrow::Table>& table ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::StringArray> complaintText,
    getStringColumn( table, complaintiq::columns::COMPLAINT_TEXT )
  );

  const std::string expectedFirstText =
    "i was charged twice and the company would not help.";

  if( complaintText->IsNull( FIRST_ROW_INDEX ) ||
      complaintText->GetString( FIRST_ROW_INDEX ) != expectedFirstText ) {
    return arrow::Status::Invalid(
      "complaint_text row 0 mismatch. Got '",
      complaintText->IsNull( FIRST_ROW_INDEX )
        ? "NULL"
        : complaintText->GetString( FIRST_ROW_INDEX ),
      "'"
    );
  }

  if( !complaintText->IsNull( SECOND_ROW_INDEX ) ) {
    return arrow::Status::Invalid(
      "complaint_text row 1 should be null, got '",
      complaintText->GetString( SECOND_ROW_INDEX ),
      "'"
    );
  }

  if( complaintText->IsNull( THIRD_ROW_INDEX ) ) {
    return arrow::Status::Invalid( "complaint_text row 2 should not be null." );
  }

  const std::string thirdText = complaintText->GetString( THIRD_ROW_INDEX );

  if( thirdText.find( "redacted_token" ) == std::string::npos ) {
    return arrow::Status::Invalid(
      "complaint_text row 2 should normalize XXXX to redacted_token, got '",
      thirdText,
      "'"
    );
  }

  return arrow::Status::OK();
}

// complaint_text_length is a simple structured feature and also verifies that
// cleaned text and blank narratives are handled consistently.
arrow::Status validateComplaintTextLength( const std::shared_ptr<arrow::Table>& table ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::StringArray> complaintText,
    getStringColumn( table, complaintiq::columns::COMPLAINT_TEXT )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Int32Array> complaintTextLength,
    getInt32Column( table, complaintiq::columns::COMPLAINT_TEXT_LENGTH )
  );

  for( int rowIndex = 0; rowIndex < EXPECTED_ROW_COUNT; rowIndex++ ) {
    const int32_t expectedLength = complaintText->IsNull( rowIndex )
      ? 0
      : static_cast<int32_t>( complaintText->GetString( rowIndex ).size() );

    const int32_t actualLength = complaintTextLength->Value( rowIndex );

    if( actualLength != expectedLength ) {
      return arrow::Status::Invalid(
        "complaint_text_length row ",
        rowIndex,
        " mismatch. Expected ",
        expectedLength,
        ", got ",
        actualLength
      );
    }
  }

  return arrow::Status::OK();
}

// The fixture feeds the ETL ISO-8601 date-time strings, so this checks that they
// were parsed down to the correct date32 values. date32 stores the number of days
// since 1970-01-01, so the expected values below are those day counts for the
// fixture's calendar dates.
arrow::Status validateDates( const std::shared_ptr<arrow::Table>& table ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Date32Array> dateReceived,
    getDate32Column( table, complaintiq::columns::DATE_RECEIVED )
  );

  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Date32Array> dateSentToCompany,
    getDate32Column( table, complaintiq::columns::DATE_SENT_TO_COMPANY )
  );

  // 2024-01-01, 2024-01-03, 2024-01-05 as days since the epoch.
  const std::vector<int32_t> expectedReceived = {19723, 19725, 19727};
  // 2024-01-02, 2024-01-04, 2024-01-06 as days since the epoch.
  const std::vector<int32_t> expectedSent = {19724, 19726, 19728};

  for( int rowIndex = 0; rowIndex < EXPECTED_ROW_COUNT; rowIndex++ ) {
    if( dateReceived->Value( rowIndex ) != expectedReceived.at( rowIndex ) ) {
      return arrow::Status::Invalid(
        "date_received row ",
        rowIndex,
        " mismatch. Expected ",
        expectedReceived.at( rowIndex ),
        ", got ",
        dateReceived->Value( rowIndex )
      );
    }

    if( dateSentToCompany->Value( rowIndex ) != expectedSent.at( rowIndex ) ) {
      return arrow::Status::Invalid(
        "date_sent_to_company row ",
        rowIndex,
        " mismatch. Expected ",
        expectedSent.at( rowIndex ),
        ", got ",
        dateSentToCompany->Value( rowIndex )
      );
    }
  }

  return arrow::Status::OK();
}

arrow::Status validateValues( const std::string& parquetPath ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Table> table,
    readParquetTable( parquetPath )
  );

  ARROW_RETURN_NOT_OK( validateRowCount( table ) );
  ARROW_RETURN_NOT_OK( validateMonetaryRelief( table ) );
  ARROW_RETURN_NOT_OK( validateHasNarrative( table ) );
  ARROW_RETURN_NOT_OK( validateComplaintText( table ) );
  ARROW_RETURN_NOT_OK( validateComplaintTextLength( table ) );
  ARROW_RETURN_NOT_OK( validateDates( table ) );

  return arrow::Status::OK();
}

int main(int argc, char* argv[]) {
  if( argc != 2 ) {
    printUsage( argv[0] );
    return 1;
  }

  const std::string parquetPath = argv[1];
  const arrow::Status status = validateValues( parquetPath );

  if( !status.ok() ) {
    std::cerr << "FAIL: " << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "PASS: Parquet values validated" << std::endl;

  return 0;  // Report success to the operating system.
}
