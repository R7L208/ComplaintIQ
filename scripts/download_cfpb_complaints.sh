#!/usr/bin/env bash

# Fail fast so a missing config variable, failed download, or failed unzip does
# not allow the script to continue and report success incorrectly.
set -euo pipefail

# Directory where the script lives: ComplaintIQ/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root: ComplaintIQ
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load shared config
# set -a automatically exports variables created by config.env. This matters
# when child processes or test mocks need access to the same configuration.
set -a
source "$PROJECT_ROOT/config.env"
set +a

# Outputs:
# - ComplaintIQ/data
# - ComplaintIQ/data/complaints.csv.zip
# - ComplaintIQ/data/complaints.csv
# Build absolute output paths from config values so the same script works from
# any launch directory and keeps project-specific paths centralized in config.env.
OUTPUT_DIR="$PROJECT_ROOT/$DATA_DIR"
OUTPUT_ZIP="$OUTPUT_DIR/$COMPLAINTS_ZIP"
OUTPUT_CSV="$OUTPUT_DIR/$COMPLAINTS_CSV"

# Create output directory if it doesn't exist.
mkdir -p "$OUTPUT_DIR"

# Avoid redownloading if the file already exists.
# Treat the extracted CSV as the source of truth. If it already exists, the
# expensive network and unzip steps can be skipped safely.
if [[ -f "$OUTPUT_CSV" ]]; then
    echo "Dataset already exists: $OUTPUT_CSV"
    ls -lh "$OUTPUT_CSV"
    exit 0
fi

echo "Downloading CFPB Consumer Complaint Database..."

# -L follows redirects, --fail turns HTTP errors into command failures, and
# --retry 3 makes the download more reliable against temporary network issues.
curl -L --fail --retry 3 "$CFPB_COMPLAINTS_URL" -o "$OUTPUT_ZIP"

echo "Unzipping file..."

# -o overwrites existing extracted files, which keeps repeated runs from pausing
# for interactive confirmation.
unzip -o "$OUTPUT_ZIP" -d "$OUTPUT_DIR"
echo "Unzipped to $OUTPUT_DIR"

echo "Removing zipped file"

# Remove the archive after extraction so the data directory only keeps the CSV
# needed by the ETL pipeline.
rm "$OUTPUT_ZIP"

echo "Done."
ls -lh "$OUTPUT_DIR"
