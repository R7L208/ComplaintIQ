#!/usr/bin/env bash

# Fail fast so a broken build, missing variable, or failed pipeline cannot be
# hidden by later commands in the smoke test.
set -euo pipefail

# Anchor the test to the repository root so it behaves the same whether it is
# launched from the repo root, a subdirectory, make, or CI.
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

# Use an isolated temp directory so the test does not depend on checked-in data
# files and does not leave generated CSV/Parquet files in the repo.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' exit

INPUT_CSV="$TMP_DIR/$COMPLAINTS_CSV"
OUTPUT_PARQUET="$TMP_DIR/$COMPLAINTS_PARQUET"

fail () {
    echo "FAIL: $1" >&2
    exit 1
}

# The quoted heredoc delimiter prevents shell expansion inside the CSV fixture.
# That keeps the test data literal if future rows contain $, backticks, or slashes.
# The fixture matches the current CFPB export header (16 columns, ISO-8601
# timestamp dates) so the smoke test runs against the real column layout.
cat > "$INPUT_CSV" << 'CSV'
Date received,Product,Sub-product,Issue,Sub-issue,Consumer complaint narrative,Company public response,Company,State,ZIP code,Tags,Submitted via,Date sent to company,Company response to consumer,Timely response?,Complaint ID
2024-01-01T08:15:30.000Z,Credit card,General-purpose credit card,Billing dispute,,I was charged twice and the company would not help.,,Test Bank,CO,80202,,Web,2024-01-02T09:00:00.000Z,Closed with monetary relief,Yes,1
2024-01-03T00:00:00.000Z,Mortgage,Conventional home mortgage,Applying for a mortgage,, ,,Example Mortgage,CA,94105,,Web,2024-01-04T00:00:00.000Z,Closed with explanation,Yes,2
CSV

echo "Starting C++ ETL Engine smoke test..."

# Build as part of the smoke test so this catches both compile/link failures and
# runtime failures from the generated binary.
make build BUILD_DIR="$BUILD_DIR"

# Verify the Makefile produced the expected executable before treating runtime
# failures as ETL bugs.
[[ -x "$ETL_BIN" ]] || fail "Expected executable not found at $ETL_BIN"

echo "Running CSV to Parquet smoke test..."

# Exercise the real command-line interface with a tiny fixture instead of calling
# internals, which makes this an end-to-end test of CSV input through Parquet output.
"$ETL_BIN" \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_PARQUET"

echo "Checking Parquet output..."

# These checks are intentionally minimal: the smoke test only proves that the ETL
# can produce a non-empty Parquet file from valid CSV input.
[[ -f "$OUTPUT_PARQUET" ]] || fail "Expected Parquet output was not created at $OUTPUT_PARQUET"
[[ -s "$OUTPUT_PARQUET" ]] || fail "Output Parquet exists at $OUTPUT_PARQUET but is empty"

echo "PASS: C++ ETL smoke test passed. \n Parquet exists at $OUTPUT_PARQUET"
