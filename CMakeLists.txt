# Require a modern CMake version so the project can use imported package targets
# from Arrow and Parquet reliably.
cmake_minimum_required(VERSION 3.25)

project(ComplaintIQ LANGUAGES CXX)

# Use C++17 because the ETL code depends on standard library features such as
# std::filesystem.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Arrow provides the in-memory columnar table format and CSV reader used by the
# ETL pipeline.
find_package(Arrow REQUIRED CONFIG)

# Parquet provides the columnar file writer used to persist the Arrow table.
find_package(Parquet REQUIRED CONFIG)

# Build a single ETL executable from the CSV-to-Parquet source file.
add_executable(complaintiq_etl
    src/etl/csv_to_parquet.cpp
)

# Build a small validator executable from the Validate-Parquet-Schema test file.
add_executable(complaintiq_validate_parquet_schema
    test/cpp/validate_parquet_schema.cpp
)

# Build another small validator executable from the Validate-Parquet-Values test file.
add_executable(complaintiq_validate_parquet_values
    test/cpp/validate_parquet_values.cpp
)

# Link against Arrow and Parquet as private implementation dependencies because
# this executable uses them internally but does not expose a library interface.
target_link_libraries(complaintiq_etl
    PRIVATE
        Arrow::arrow_shared
        Parquet::parquet_shared
)
target_link_libraries(complaintiq_validate_parquet_schema
    PRIVATE
        Arrow::arrow_shared
        Parquet::parquet_shared
)
target_link_libraries(complaintiq_validate_parquet_values
    PRIVATE
        Arrow::arrow_shared
        Parquet::parquet_shared
)

# Expose the shared include/ tree so every target can resolve the schema
# contract header via #include "complaintiq/schema.h".
target_include_directories(complaintiq_etl
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(complaintiq_validate_parquet_schema
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(complaintiq_validate_parquet_values
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)
