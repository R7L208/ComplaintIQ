/*
 *  validate_parquet_schema.cpp
 *
 *  Small validator executable that opens the Parquet file produced by the ETL
 *  and checks that its column count, column names, column types, and row count
 *  match the expected schema contract.
 */

// Project headers: the shared schema contract defines the expected columns.
#include "complaintiq/schema.h"  // for complaintiq::ExpectedColumn and getExpectedSchema

// Third-party libraries: Apache Arrow and Parquet provide the columnar table
// format and the Parquet reader used to inspect the ETL output.
#include <arrow/api.h>              // for core Arrow types like Table and Field
#include <arrow/io/api.h>           // for Arrow file input streams
#include <parquet/arrow/reader.h>   // for reading Parquet files into Arrow

#include <iostream>  // for std::cerr and std::cout messages
#include <memory>    // for std::shared_ptr and std::unique_ptr
#include <string>    // for std::string
#include <vector>    // for std::vector of expected columns

#include <cstdint>  // for int64_t

// The expected row count is constant because the schema test creates a fixed
// three-row CSV fixture before running the ETL.
const int64_t EXPECTED_ROW_COUNT = 3;

void printUsage( const char* pProgramName ) {
  std::cerr << "Usage: " << pProgramName
            << " [--print-schema | --no-row-count] <parquet_file_path>" << std::endl;
}

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

  return table;
}

// Check the count first so missing derived fields or unexpected raw CFPB
// columns fail before the more detailed name and type checks run.
arrow::Status validateColumnCount(
  const std::shared_ptr<arrow::Table>& table,
  const std::vector<complaintiq::ExpectedColumn>& expectedSchema
) {
  const int actualColumnCount = table->num_columns();
  const int expectedColumnCount = static_cast<int>( expectedSchema.size() );

  if( actualColumnCount != expectedColumnCount ) {
    return arrow::Status::Invalid(
      "Expected ",
      expectedColumnCount,
      " columns, got ",
      actualColumnCount
    );
  }

  return arrow::Status::OK();
}

// Require a stable column order so downstream notebooks and model code can
// depend on deterministic Parquet output.
arrow::Status validateColumnNamesAndTypes(
  const std::shared_ptr<arrow::Table>& table,
  const std::vector<complaintiq::ExpectedColumn>& expectedSchema
) {
  for( int columnIndex = 0;
       columnIndex < static_cast<int>( expectedSchema.size() );
       columnIndex++ ) {
    const complaintiq::ExpectedColumn& expectedColumn = expectedSchema.at( columnIndex );
    std::shared_ptr<arrow::Field> actualField = table->field( columnIndex );

    if( actualField->name() != expectedColumn.name ) {
      return arrow::Status::Invalid(
        "Column ",
        columnIndex,
        " name mismatch. Expected '",
        expectedColumn.name,
        "', got '",
        actualField->name(),
        "'"
      );
    }

    // Validate types so ML labels and engineered features are not accidentally
    // emitted as strings during CSV conversion.
    if( !actualField->type()->Equals( expectedColumn.type ) ) {
      return arrow::Status::Invalid(
        "Column '",
        expectedColumn.name,
        "' type mismatch. Expected '",
        expectedColumn.type->ToString(),
        "', got '",
        actualField->type()->ToString(),
        "'"
      );
    }
  }

  return arrow::Status::OK();
}

// The schema fixture has three rows, and the default ETL path should keep every
// row. Narrative-only filtering should be covered by a separate test.
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

// checkRowCount is only meaningful for the fixed test fixture. When validating
// the real ETL output, which has millions of rows, the caller passes false so
// only the column count, names, and types are checked against the contract.
arrow::Status validateSchema( const std::string& parquetPath, bool checkRowCount ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Table> table,
    readParquetTable( parquetPath )
  );

  const std::vector<complaintiq::ExpectedColumn> expectedSchema = complaintiq::getExpectedSchema();

  ARROW_RETURN_NOT_OK( validateColumnCount( table, expectedSchema ) );
  ARROW_RETURN_NOT_OK( validateColumnNamesAndTypes( table, expectedSchema ) );

  if( checkRowCount ) {
    ARROW_RETURN_NOT_OK( validateRowCount( table ) );
  }

  return arrow::Status::OK();
}

// Print each column as "name: type" so a shell test can assert the physical
// Arrow types independently of the shared getExpectedSchema() contract.
arrow::Status printSchema( const std::string& parquetPath ) {
  ARROW_ASSIGN_OR_RAISE(
    std::shared_ptr<arrow::Table> table,
    readParquetTable( parquetPath )
  );

  const std::shared_ptr<arrow::Schema> schema = table->schema();

  for( const std::shared_ptr<arrow::Field>& field : schema->fields() ) {
    std::cout << field->name() << ": " << field->type()->ToString() << std::endl;
  }

  return arrow::Status::OK();
}

int main(int argc, char* argv[]) {
  // The optional --print-schema flag dumps the Parquet column types instead of
  // validating them, so tests can check types without trusting the shared schema.
  if( argc == 3 && std::string( argv[1] ) == "--print-schema" ) {
    const arrow::Status status = printSchema( argv[2] );

    if( !status.ok() ) {
      std::cerr << "FAIL: " << status.ToString() << std::endl;
      return 1;
    }

    return 0;  // Report success to the operating system.
  }

  // --no-row-count validates the column contract but skips the fixed row-count
  // check, which is what the real ETL output needs since it has millions of rows.
  bool checkRowCount = true;
  std::string parquetPath;

  if( argc == 3 && std::string( argv[1] ) == "--no-row-count" ) {
    checkRowCount = false;
    parquetPath = argv[2];
  } else if( argc == 2 ) {
    parquetPath = argv[1];
  } else {
    printUsage( argv[0] );
    return 1;
  }

  const arrow::Status status = validateSchema( parquetPath, checkRowCount );

  if( !status.ok() ) {
    std::cerr << "FAIL: " << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "PASS: Parquet schema validated" << std::endl;

  return 0;  // Report success to the operating system.
}
