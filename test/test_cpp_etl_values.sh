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
VALUES_VALIDATOR_BIN="$BUILD_DIR/$VALUES_VALIDATOR_BIN_NAME"

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

# This fixture uses the current CFPB export shape (16 columns, ISO-8601 timestamp
# dates) and intentionally covers the first ETL value cases:
#   - monetary relief should become label 1
#   - blank narrative should become null with has_narrative=false
#   - XXXX-style text should be normalized for text modeling
#   - the ISO date-time strings should be parsed down to date32 day values
cat > "$INPUT_CSV" <<'CSV'
Date received,Product,Sub-product,Issue,Sub-issue,Consumer complaint narrative,Company public response,Company,State,ZIP code,Tags,Submitted via,Date sent to company,Company response to consumer,Timely response?,Complaint ID
2024-01-01T08:15:30.000Z,Credit card,General-purpose credit card,Billing dispute,,I was charged twice and the company would not help.,,Test Bank,CO,80202,,Web,2024-01-02T09:00:00.000Z,Closed with monetary relief,Yes,1
2024-01-03T00:00:00.000Z,Mortgage,Conventional home mortgage,Applying for a mortgage,, ,,Example Mortgage,CA,94105,,Web,2024-01-04T00:00:00.000Z,Closed with explanation,Yes,2
2024-01-05T00:00:00.000Z,Checking or savings account,Checking account,Managing an account,,My account had XXXX fees and duplicate charges.,,Another Bank,NY,10001,Older American,Phone,2024-01-06T00:00:00.000Z,Closed with non-monetary relief,No,3
CSV

echo "Building C++ ETL engine..."
make build BUILD_DIR="$BUILD_DIR"

echo "Building C++ values validator..."
cmake --build "$BUILD_DIR" --target "$VALUES_VALIDATOR_BIN_NAME"

[[ -x "$ETL_BIN" ]] || fail "Expected ETL executable not found at $ETL_BIN"
[[ -x "$VALUES_VALIDATOR_BIN" ]] || fail "Expected values validator not found at $VALUES_VALIDATOR_BIN"

echo "Running ETL..."
"$ETL_BIN" \
  --input "$INPUT_CSV" \
  --output "$OUTPUT_PARQUET"

[[ -s "$OUTPUT_PARQUET" ]] || fail "Expected non-empty Parquet output at $OUTPUT_PARQUET"

# Validate with the C++ validator so the test uses only the Arrow/Parquet stack
# already required by the ETL engine.
echo "Validating Parquet values..."
"$VALUES_VALIDATOR_BIN" "$OUTPUT_PARQUET"

echo "PASS: C++ ETL values test passed"
