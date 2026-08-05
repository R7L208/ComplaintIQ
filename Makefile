# ComplaintIQ
#
# CFPB consumer complaint ETL pipeline. Downloads the CFPB complaints dataset
# and converts the CSV into cleaned Parquet using a C++ (Apache Arrow) engine.
# This Makefile orchestrates the data download, C++ build, Parquet conversion,
# and test targets.

# Load shared project variables so Make targets use the same paths and filenames
# as the shell scripts instead of duplicating configuration.
include config.env

# Optional, git-ignored local overrides (e.g. DATABRICKS_PYPI_PROXY). See
# local.mk.example. The leading dash means "don't error if it doesn't exist".
-include local.mk

# Path to the script that downloads and extracts the CFPB complaints CSV.
DOWNLOAD_SCRIPT := scripts/download_cfpb_complaints.sh

# Treat the extracted CSV as the download artifact because the download script
# removes the ZIP file after extraction.
COMPLAINTS_CSV_PATH := $(DATA_DIR)/$(COMPLAINTS_CSV)

# Parquet output paths derived from config.env. The output file names can be
# overridden from the command line without editing the Makefile.
COMPLAINTS_PARQUET_PATH := $(DATA_DIR)/$(COMPLAINTS_PARQUET)
COMPLAINTS_NARRATIVE_ONLY_PARQUET_PATH := $(DATA_DIR)/$(COMPLAINTS_NARRATIVE_ONLY_PARQUET)

# Keep C++ build outputs separate from source files so generated artifacts are
# easy to clean and do not clutter the project root. BUILD_DIR comes from
# config.env and can be overridden from the command line.
ETL_BIN := $(BUILD_DIR)/$(ETL_BIN_NAME)
SCHEMA_VALIDATOR_BIN := $(BUILD_DIR)/$(SCHEMA_VALIDATOR_BIN_NAME)

# Optional dependency locations for Apache Arrow / Parquet.
# vcpkg is preferred when available because it is cross-platform; Homebrew is a
# convenient fallback for local macOS development.
VCPKG_ROOT ?= $(HOME)/vcpkg
VCPKG_TOOLCHAIN_FILE := $(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake
ARROW_PREFIX ?= $(shell brew --prefix apache-arrow 2>/dev/null)

CMAKE_BUILD_TYPE ?= Release
CMAKE_CONFIG_ARGS := -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Python environment (uv). VENV_DIR / PYTHON_VERSION come from config.env.
# The interpreter path and an install stamp let Make skip reinstalling when the
# environment is already provisioned. requirements.txt is the committed
# lockfile that pins exact versions so the venv is reproducible across machines.
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_STAMP := $(VENV_DIR)/.install-stamp
REQUIREMENTS := requirements.txt

# EXTRAS selects optional dependency groups from pyproject.toml. The `local`
# extra (pyspark, pyarrow, ipykernel) is installed by default so the notebooks
# run locally -- those are package extras, not hard deps, because Databricks
# serverless already provides them. Add more, e.g.:
#   make venv EXTRAS="local boost"   -> also installs the [boost] extra
#   make venv EXTRAS="local dev"     -> also installs formatting/type-check tools
EXTRAS ?= local
EXTRAS_ARGS := $(foreach extra,$(EXTRAS),--extra $(extra))

# Private-index fallback for machines that cannot reach public PyPI (e.g. a
# supply-chain hosts blocklist). When a target that needs an index is run and
# pypi.org is unreachable, fall back to a developer-configured internal index
# (DATABRICKS_PYPI_PROXY, set in the git-ignored local.mk or the environment).
# This is a no-op unless that proxy is set AND pypi.org is actually down, so
# machines with normal PyPI access and external users are unaffected. uv reads
# the exported UV_DEFAULT_INDEX; requirements.txt does not record the index URL
# (uv omits it by default), so the private index never leaks into the lockfile.
# Export UV_DEFAULT_INDEX yourself to override the probe entirely.
NETWORK_GOALS := lock venv
ifneq ($(filter $(NETWORK_GOALS),$(MAKECMDGOALS)),)
ifndef UV_DEFAULT_INDEX
ifneq ($(strip $(DATABRICKS_PYPI_PROXY)),)
ifneq ($(shell curl -fsS -m 3 -o /dev/null https://pypi.org/simple/ 2>/dev/null && echo up),up)
export UV_DEFAULT_INDEX := $(DATABRICKS_PYPI_PROXY)
$(info make: public PyPI unreachable -- falling back to the developer-configured index.)
endif
endif
endif
endif

# Add the dependency location CMake needs, but only when that dependency manager
# actually exists on the machine running the build.
ifneq ($(wildcard $(VCPKG_TOOLCHAIN_FILE)),)
CMAKE_CONFIG_ARGS += -DCMAKE_TOOLCHAIN_FILE=$(VCPKG_TOOLCHAIN_FILE)
else ifneq ($(ARROW_PREFIX),)
CMAKE_CONFIG_ARGS += -DCMAKE_PREFIX_PATH=$(ARROW_PREFIX)
endif

# NARRATIVE_ONLY=1 switches the ETL output to a filtered Parquet file without
# requiring a separate Make target.
NARRATIVE_ONLY ?= 0
CLEAN ?= build

ifeq ($(NARRATIVE_ONLY),1)
ETL_FLAGS := --require-narrative
ETL_OUTPUT_PATH := $(COMPLAINTS_NARRATIVE_ONLY_PARQUET_PATH)
else
ETL_FLAGS :=
ETL_OUTPUT_PATH := $(COMPLAINTS_PARQUET_PATH)
endif

all: parquet

# Make can skip the download when the expected CSV already exists.
download_data: $(COMPLAINTS_CSV_PATH)

# Re-run the downloader if either the script or config changes.
$(COMPLAINTS_CSV_PATH): $(DOWNLOAD_SCRIPT) config.env
	@echo "Downloading CFPB Complaints dataset..."
	@echo "Using script: $(DOWNLOAD_SCRIPT)"
	@chmod +x $(DOWNLOAD_SCRIPT)
	@$(DOWNLOAD_SCRIPT)

build:
	@echo "Building C++ ETL engine..."
	@cmake -S . -B $(BUILD_DIR) $(CMAKE_CONFIG_ARGS)
	@cmake --build $(BUILD_DIR) --target $(ETL_BIN_NAME)

parquet: build download_data
	@echo "Converting CSV to cleaned Parquet..."
	@mkdir -p $(DATA_DIR)
	@$(ETL_BIN) \
		--input $(COMPLAINTS_CSV_PATH) \
		--output $(ETL_OUTPUT_PATH) \
		$(ETL_FLAGS)
	@$(MAKE) validate-parquet PARQUET_PATH=$(ETL_OUTPUT_PATH)

# Check that a produced Parquet actually matches the snake_case schema contract
# (column count, names, and types). This runs automatically after 'make parquet'
# so a bad or stale file is caught here instead of when we start modeling later.
# Row count is skipped because the real dataset has millions of rows.
# PARQUET_PATH defaults to the standard output but can be overridden to point at
# any Parquet file, e.g. the narrative-only build.
PARQUET_PATH ?= $(ETL_OUTPUT_PATH)
validate-parquet:
	@echo "Validating $(PARQUET_PATH) against the schema contract..."
	@cmake --build $(BUILD_DIR) --target $(SCHEMA_VALIDATOR_BIN_NAME)
	@$(SCHEMA_VALIDATOR_BIN) --no-row-count $(PARQUET_PATH)

# --- Python environment for notebook / ML work in Zed (uv) ---------------

# Compile pyproject.toml into a fully pinned lockfile. Re-runs whenever
# pyproject.toml changes. Commit requirements.txt so the venv is reproducible.
lock: $(REQUIREMENTS)

$(REQUIREMENTS): pyproject.toml
	@echo "Locking Python dependencies -> $(REQUIREMENTS)..."
	@uv pip compile pyproject.toml $(EXTRAS_ARGS) --output-file $(REQUIREMENTS)

# Create the uv-managed virtual environment and install the locked
# dependencies (including ipykernel, so Zed can attach a Jupyter kernel).
# Rebuilds when the lockfile changes; otherwise the stamp lets Make no-op.
venv: $(VENV_STAMP)

$(VENV_STAMP): $(REQUIREMENTS)
	@echo "Creating uv virtual environment (Python $(PYTHON_VERSION)) in $(VENV_DIR)..."
	@uv venv --python $(PYTHON_VERSION) $(VENV_DIR)
	@echo "Installing locked dependencies into $(VENV_DIR)..."
	@uv pip sync --python $(VENV_PYTHON) $(REQUIREMENTS)
	@touch $(VENV_STAMP)
	@echo "Virtual environment ready."
	@echo "In Zed: open a .ipynb (or use the REPL) and select the $(VENV_DIR) kernel."

clean_venv:
	@echo "Removing Python virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Successfully removed $(VENV_DIR)"

# Register a global Jupyter kernelspec pointing at this venv. Zed and VS Code
# auto-detect the venv directly and do NOT need this, but JupyterLab / classic
# Notebook / nbconvert discover kernels from the global list, so register there.
# Remove later with: jupyter kernelspec uninstall complaintiq
kernel: $(VENV_STAMP)
	@echo "Registering global Jupyter kernel 'complaintiq'..."
	@$(VENV_PYTHON) -m ipykernel install --user \
		--name complaintiq --display-name "Python (ComplaintIQ)"

figures:
	@echo "Regenerating result figures into docs/figures/ ..."
	@$(VENV_PYTHON) scripts/generate_figures.py

test:
	@echo "Running test suite..."
	@bash test/test_download_cfpb_complaints.sh
	@bash test/test_cpp_etl_smoke.sh
	@bash test/test_cpp_etl_schema.sh
	@bash test/test_cpp_etl_values.sh
	@bash test/test_cpp_etl_types.sh

clean_data:
	@echo "Removing downloaded data..."
	@rm -rf $(DATA_DIR)
	@echo "Successfully removed data"

# CLEAN selects the cleanup scope:
#   make clean              removes build artifacts
#   make clean CLEAN=data   removes downloaded/generated data
#   make clean CLEAN=venv   removes the Python virtual environment
#   make clean CLEAN=all    removes build artifacts, data, and the venv
clean:
ifeq ($(CLEAN),build)
	@echo "Removing C++ build artifacts..."
	@rm -rf $(BUILD_DIR)
else ifeq ($(CLEAN),data)
	@echo "Removing downloaded and generated data..."
	@rm -rf $(DATA_DIR)
else ifeq ($(CLEAN),venv)
	@echo "Removing Python virtual environment..."
	@rm -rf $(VENV_DIR)
else ifeq ($(CLEAN),all)
	@echo "Removing C++ build artifacts, data, and the Python venv..."
	@rm -rf $(BUILD_DIR) $(DATA_DIR) $(VENV_DIR)
else
	$(error Invalid CLEAN value: $(CLEAN). Use CLEAN=build, CLEAN=data, CLEAN=venv, or CLEAN=all)
endif

# --- Databricks (Free Edition): bundle + bi-directional Git folder sync ---
# Bi-directional sync runs through git: the Databricks Git folder and your local
# clone both push/pull to GitHub, so git is the sync medium. The bundle
# (databricks.yml) manages cloud resources: the UC volume and a serverless job.
#
# DATABRICKS_PROFILE is personal and comes from the git-ignored local.mk. It
# supplies your workspace host, so every developer targets their own workspace.
DB := databricks
DB_PROFILE_FLAG := $(if $(DATABRICKS_PROFILE),--profile $(DATABRICKS_PROFILE),)

# Fail fast, with a helpful message, when the personal profile is not configured.
define require-db-profile
@if [ -z "$(DATABRICKS_PROFILE)" ]; then \
	echo "DATABRICKS_PROFILE is not set. Copy local.mk.example to local.mk and set it."; \
	exit 1; \
fi
endef

# Full Git folder path is derived from the signed-in CLI user at run time, so no
# personal identity is committed. Lazily evaluated (only when a db-* target runs).
DB_USER = $(shell $(DB) current-user me $(DB_PROFILE_FLAG) 2>/dev/null | \
	python3 -c "import sys, json; print(json.load(sys.stdin)['userName'])" 2>/dev/null)
DATABRICKS_GIT_FOLDER = /Users/$(DB_USER)/$(DATABRICKS_GIT_FOLDER_NAME)
GIT_BRANCH = $(shell git branch --show-current)
VOLUME_PATH := dbfs:/Volumes/$(UC_CATALOG)/$(UC_SCHEMA)/$(UC_VOLUME)

# Clone URL for the Git folder, derived from the local 'origin' remote so a fork
# targets its own repo. Databricks Git folders need HTTPS, so convert the
# scp-style SSH form (git@host:owner/repo.git) to https://host/owner/repo.git.
DATABRICKS_REPO_URL = $(shell git remote get-url origin 2>/dev/null | \
	sed -E 's|^git@([^:]+):|https://\1/|; s|^ssh://git@([^/]+)/|https://\1/|')

# One-time: create the Databricks Git folder that clones the repo. Requires a
# GitHub credential linked in Databricks (Settings > Linked accounts > Git).
db-repo-create:
	$(require-db-profile)
	@echo "Creating Databricks Git folder $(DATABRICKS_GIT_FOLDER)..."
	@$(DB) repos create $(DATABRICKS_REPO_URL) gitHub \
		--path $(DATABRICKS_GIT_FOLDER) $(DB_PROFILE_FLAG)

# Pull the latest pushed commit into the Databricks Git folder. Run this after a
# local push so the Databricks side updates without clicking in the UI.
db-pull:
	$(require-db-profile)
	@echo "Pulling latest $(GIT_BRANCH) into the Databricks Git folder..."
	@$(DB) repos update $(DATABRICKS_GIT_FOLDER) \
		--branch $(GIT_BRANCH) $(DB_PROFILE_FLAG)

# Validate / deploy the bundle resources (UC volume + serverless job).
db-validate:
	$(require-db-profile)
	@$(DB) bundle validate $(DB_PROFILE_FLAG)

db-deploy:
	$(require-db-profile)
	@echo "Deploying bundle resources (volume + job)..."
	@$(DB) bundle deploy $(DB_PROFILE_FLAG)

# Run the exploration notebook as a serverless job.
db-run:
	$(require-db-profile)
	@$(DB) bundle run complaintiq_explore $(DB_PROFILE_FLAG)

# Upload the generated Parquet(s) to the UC volume. Large data never goes in git;
# run 'make parquet' first to produce it locally. The narrative-only file is
# optional -- it is uploaded too when present (build it with
# 'make parquet NARRATIVE_ONLY=1'), but its absence is not an error.
db-data:
	$(require-db-profile)
	@test -f $(COMPLAINTS_PARQUET_PATH) || { \
		echo "$(COMPLAINTS_PARQUET_PATH) not found. Run 'make parquet' first."; exit 1; }
	@echo "Uploading $(COMPLAINTS_PARQUET) to $(VOLUME_PATH)/ ..."
	@$(DB) fs cp $(COMPLAINTS_PARQUET_PATH) \
		$(VOLUME_PATH)/$(COMPLAINTS_PARQUET) --overwrite $(DB_PROFILE_FLAG)
	@if [ -f $(COMPLAINTS_NARRATIVE_ONLY_PARQUET_PATH) ]; then \
		echo "Uploading $(COMPLAINTS_NARRATIVE_ONLY_PARQUET) to $(VOLUME_PATH)/ ..."; \
		$(DB) fs cp $(COMPLAINTS_NARRATIVE_ONLY_PARQUET_PATH) \
			$(VOLUME_PATH)/$(COMPLAINTS_NARRATIVE_ONLY_PARQUET) --overwrite $(DB_PROFILE_FLAG); \
	else \
		echo "Skipping $(COMPLAINTS_NARRATIVE_ONLY_PARQUET) (not built; run 'make parquet NARRATIVE_ONLY=1')."; \
	fi

# Mark these as command targets instead of real files so make always runs their
# recipes when requested.
.PHONY: all download_data build parquet validate-parquet figures test clean_data clean lock venv clean_venv kernel \
	db-repo-create db-pull db-validate db-deploy db-run db-data
