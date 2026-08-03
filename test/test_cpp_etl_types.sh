#!/usr/bin/env bash

set -euo pipefail

# Resolve the project root from this file so the test works no matter where the
# script is called from.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load shared project config. Capture any BUILD_DIR override first so a value
# passed in the environment still wins after sourcing the defaults.
build_dir_override="${BUILD_DIR:-}"
set -a
source "$PROJECT_ROOT/config.env"
set +a
BUILD_DIR="${build_dir_override:-$BUILD_DIR}"

ETL_BIN="$BUILD_DIR/$ETL_BIN_NAME"
SCHEMA_VALIDATOR_BIN="$BUILD_DIR/$SCHEMA_VALIDATOR_BIN_NAME"

# Use a temporary dataset so this test never depends on the real CFPB download
# or modifies files in data/.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

INPUT_CSV="$TMP_DIR/$COMPLAINTS_CSV"
OUTPUT_PARQUET="$TMP_DIR/$COMPLAINTS_PARQUET"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

# These are the physical Arrow types each output column must have, spelled the
# way arrow::DataType::ToString() renders them. They are asserted independently
# of the shared schema contract so a wrong type in getExpectedSchema() cannot
# hide behind the ETL and validator agreeing with each other.
assert_type() {
  local columnName="$1"
  local expectedType="$2"

  grep -Fxq "$columnName: $expectedType" <<< "$SCHEMA_DUMP" \
    || fail "Column '$columnName' should be '$expectedType' but the Parquet schema is:
$SCHEMA_DUMP"
}

# The fixture matches the current CFPB export header (16 columns, ISO-8601
# timestamp dates) and is chosen to exercise type coercion: an integer id, the
# ISO date-time strings that must become date32, Yes/No response flags, and a
# leading-zero ZIP that must survive as a string rather than being read as a number.
cat > "$INPUT_CSV" <<'CSV'
Date received,Product,Sub-product,Issue,Sub-issue,Consumer complaint narrative,Company public response,Company,State,ZIP code,Tags,Submitted via,Date sent to company,Company response to consumer,Timely response?,Complaint ID
2024-01-01T08:15:30.000Z,Credit card,General-purpose credit card,Billing dispute,,I was charged twice and the company would not help.,,Test Bank,CO,80202,,Web,2024-01-02T09:00:00.000Z,Closed with monetary relief,Yes,1
2024-01-03T00:00:00.000Z,Mortgage,Conventional home mortgage,Applying for a mortgage,, ,,Example Mortgage,MA,01001,,Web,2024-01-04T00:00:00.000Z,Closed with explanation,No,2
2024-01-05T00:00:00.000Z,Checking or savings account,Checking account,Managing an account,,My account had XXXX fees and duplicate charges.,,Another Bank,NY,10001,Older American,Phone,2024-01-06T00:00:00.000Z,Closed with non-monetary relief,Yes,3
CSV

echo "Building C++ ETL engine..."
make build BUILD_DIR="$BUILD_DIR"

echo "Building C++ schema validator..."
cmake --build "$BUILD_DIR" --target "$SCHEMA_VALIDATOR_BIN_NAME"

[[ -x "$ETL_BIN" ]] || fail "Expected ETL executable not found at $ETL_BIN"
[[ -x "$SCHEMA_VALIDATOR_BIN" ]] || fail "Expected schema validator not found at $SCHEMA_VALIDATOR_BIN"

echo "Running ETL..."
"$ETL_BIN" \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_PARQUET"

[[ -s "$OUTPUT_PARQUET" ]] || fail "Expected non-empty Parquet output at $OUTPUT_PARQUET"

# Dump the actual Parquet column types once, then assert each one below.
echo "Checking Parquet column types..."
SCHEMA_DUMP="$( "$SCHEMA_VALIDATOR_BIN" --print-schema "$OUTPUT_PARQUET" )"

assert_type "complaint_id" "int64"
assert_type "date_received" "date32[day]"
assert_type "date_sent_to_company" "date32[day]"
assert_type "product" "string"
assert_type "sub_product" "string"
assert_type "issue" "string"
assert_type "sub_issue" "string"
assert_type "complaint_text" "string"
assert_type "company" "string"
assert_type "state" "string"
assert_type "zip_code" "string"
assert_type "tags" "string"
assert_type "submitted_via" "string"
assert_type "timely_response" "bool"
assert_type "company_response_to_consumer" "string"
assert_type "complaint_text_length" "int32"
assert_type "has_narrative" "bool"
assert_type "monetary_relief" "int8"

echo "PASS: C++ ETL types test passed"
