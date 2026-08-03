#!/usr/bin/env bash

# Fail fast so a missing variable, failed assertion, or failed command cannot be
# hidden by later test steps.
set -euo pipefail

# Anchor paths to the real repository so the tests behave the same when launched
# from the repo root, another directory, make, or CI.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Put fake test binaries first in PATH later so curl/unzip can be tested without
# making a real network request or depending on the system unzip behavior.
FAKE_BIN_DIR="$PROJECT_ROOT/test/scripts"
SCRIPT_REL_PATH="scripts/download_cfpb_complaints.sh"

fail () {
    echo "FAIL: $*" >&2
    exit 1
}

assert_file_exists () {
    [[ -f "$1" ]] || fail "Expected file to exist: $1"
}

assert_file_not_exists () {
    [[ ! -e "$1" ]] || fail "Expected file to not exist: $1"
}

assert_contains () {
    local file="$1"
    local expected="$2"

    assert_file_exists "$file"

    # grep -F treats the expected text as a literal string, and -- prevents
    # values beginning with '-' from being interpreted as grep options.
    grep -Fq -- "$expected" "$file" || {
        echo "----- $file contents -----" >&2
        cat "$file" >&2
        echo "--------------------------" >&2
        fail "Expected $file to contain: $expected"
    }
}

# Use one temporary root for all test cases so each test can get an isolated
# fake project directory while still sharing automatic cleanup.
TMP_ROOT="$(mktemp -d)"
# Clean up on success or failure.
trap 'rm -rf "$TMP_ROOT"' EXIT

TEST_NUM=0
TMP_DIR=""

append_test_config_overrides () {
      # The quoted heredoc keeps these test values literal and prevents the shell
      # from expanding anything inside the config block.
      cat >> "$TMP_DIR/config.env" <<'EOF'

    # Test overrides
    CFPB_COMPLAINTS_URL=https://files.consumerfinance.gov/ccdb/complaints.csv.zip
    DATA_DIR=data
    COMPLAINTS_ZIP=complaints.csv.zip
    COMPLAINTS_CSV=complaints.csv
EOF
}

setup_fake_dir() {
  TEST_NUM=$((TEST_NUM + 1))
  TMP_DIR="$TMP_ROOT/dir_$TEST_NUM"

  mkdir -p "$TMP_DIR/scripts"

  # Copy only the files needed by the downloader so each test runs against a
  # small fake repository instead of mutating the real project tree.
  cp "$PROJECT_ROOT/config.env" "$TMP_DIR/config.env"
  cp "$PROJECT_ROOT/$SCRIPT_REL_PATH" "$TMP_DIR/$SCRIPT_REL_PATH"

  if [[ -f "$PROJECT_ROOT/Makefile" ]]; then
    cp "$PROJECT_ROOT/Makefile" "$TMP_DIR/Makefile"
  fi

  chmod +x "$TMP_DIR/$SCRIPT_REL_PATH"

  append_test_config_overrides
}

run_downloader_from_dir_root() {
    (
        cd "$TMP_DIR"

        # PATH injects fake curl/unzip commands, while TEST_* variables give
        # those fake commands a place to write logs and generated files.
        PATH="$FAKE_BIN_DIR:$PATH" \
        TEST_DIR_ROOT="$TMP_DIR" \
        TEST_CURL_LOG="$TMP_DIR/curl.log" \
        TEST_UNZIP_LOG="$TMP_DIR/unzip.log" \
        bash "$SCRIPT_REL_PATH"
    )
}

run_downloader_from_scripts_dir() {
    (
        cd "$TMP_DIR/scripts"

        # Running inside a subshell keeps the directory and environment changes
        # local to this helper instead of affecting later tests.
        PATH="$FAKE_BIN_DIR:$PATH" \
        TEST_DIR_ROOT="$TMP_DIR" \
        TEST_CURL_LOG="$TMP_DIR/curl.log" \
        TEST_UNZIP_LOG="$TMP_DIR/unzip.log" \
        bash "./download_cfpb_complaints.sh"
    )
}

test_download_creates_data_dir_unzips_and_removes_zip() {
    setup_fake_dir
    source "$TMP_DIR/config.env"

    local expected_data_dir="$TMP_DIR/$DATA_DIR"
    local expected_zip="$expected_data_dir/$COMPLAINTS_ZIP"
    local expected_csv="$expected_data_dir/$COMPLAINTS_CSV"

    # Capture stdout/stderr so failures can print the downloader output only when
    # it is useful for debugging.
    run_downloader_from_dir_root > "$TMP_DIR/stdout.log" 2>&1 || {
        cat "$TMP_DIR/stdout.log" >&2
        fail "Downloader failed"
    }

    assert_file_exists "$expected_csv"
    assert_file_not_exists "$expected_zip"

    # The fake curl/unzip logs prove the downloader passed the expected paths to
    # external tools without requiring real downloads or archive extraction.
    assert_contains "$TMP_DIR/curl.log" "curl_output=$expected_zip"
    assert_contains "$TMP_DIR/unzip.log" "unzip_zip=$expected_zip"
    assert_contains "$TMP_DIR/unzip.log" "unzip_dir=$expected_data_dir"
}

test_existing_csv_skips_download_and_unzip() {
    setup_fake_dir
    source "$TMP_DIR/config.env"

    mkdir -p "$TMP_DIR/$DATA_DIR"
    echo "already exists" > "$TMP_DIR/$DATA_DIR/$COMPLAINTS_CSV"

    run_downloader_from_dir_root > "$TMP_DIR/stdout.log" 2>&1 || {
        cat "$TMP_DIR/stdout.log" >&2
        fail "Downloader failed"
    }

    # This verifies the downloader is idempotent: if the CSV already exists, it
    # should not waste time downloading or unzipping again.
    assert_file_exists "$TMP_DIR/$DATA_DIR/$COMPLAINTS_CSV"
    assert_file_not_exists "$TMP_DIR/curl.log"
    assert_file_not_exists "$TMP_DIR/unzip.log"
    assert_contains "$TMP_DIR/stdout.log" "Dataset already exists"
}

test_download_uses_url_from_config() {
    setup_fake_dir

    local custom_url="https://example.test/custom-complaints.csv.zip"

    # Override the URL after the default test config so this test proves the
    # downloader reads configuration instead of hard-coding the CFPB URL.
    cat >> "$TMP_DIR/config.env" <<EOF
    CFPB_COMPLAINTS_URL=$custom_url
EOF

    run_downloader_from_dir_root > "$TMP_DIR/stdout.log" 2>&1 || {
        cat "$TMP_DIR/stdout.log" >&2
        fail "Downloader failed"
    }

    assert_contains "$TMP_DIR/curl.log" "$custom_url"
}

test_script_works_when_run_from_scripts_directory() {
    setup_fake_dir
    source "$TMP_DIR/config.env"

    local expected_data_dir="$TMP_DIR/$DATA_DIR"
    local expected_zip="$expected_data_dir/$COMPLAINTS_ZIP"
    local expected_csv="$expected_data_dir/$COMPLAINTS_CSV"

    # This catches path bugs where the downloader only works from the repository
    # root but fails when launched from its own scripts directory.
    run_downloader_from_scripts_dir > "$TMP_DIR/stdout.log" 2>&1 || {
        cat "$TMP_DIR/stdout.log" >&2
        fail "Downloader failed"
    }
    assert_file_exists "$expected_csv"
    assert_file_not_exists "$expected_zip"

    assert_contains "$TMP_DIR/curl.log" "curl_output=$expected_zip"
    assert_contains "$TMP_DIR/unzip.log" "unzip_dir=$expected_data_dir"
}

run_test() {
    local test_name="$1"

    echo "Running $test_name..."
    "$test_name"
    echo "PASS: $test_name"
}

run_test test_download_creates_data_dir_unzips_and_removes_zip
run_test test_existing_csv_skips_download_and_unzip
run_test test_download_uses_url_from_config
run_test test_script_works_when_run_from_scripts_directory

echo "All download script tests passed."
