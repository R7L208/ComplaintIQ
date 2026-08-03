/*
 *  schema.h
 *
 *  Single source of truth for the ETL output-schema contract shared between the
 *  CSV-to-Parquet ETL and the Parquet validators. It defines the output column
 *  names, the raw CFPB source column name used on the input side, and the
 *  expected Arrow schema (column names paired with their Arrow types).
 */

#ifndef COMPLAINTIQ_SCHEMA_H
#define COMPLAINTIQ_SCHEMA_H

// Third-party libraries: Apache Arrow provides the DataType helpers (utf8,
// int32, boolean, int8) used to describe the expected output column types.
#include <arrow/api.h>  // for arrow::DataType and the type factory functions

#include <memory>   // for std::shared_ptr
#include <string>   // for std::string
#include <vector>   // for std::vector of expected columns

namespace complaintiq {

// The output column names are centralized here so the ETL and both validators
// reference the same string literals instead of duplicating them.
namespace columns {

// Using inline const std::string keeps a single definition across every
// translation unit that includes this header under C++17.
inline const std::string COMPLAINT_ID = "complaint_id";
inline const std::string DATE_RECEIVED = "date_received";
inline const std::string DATE_SENT_TO_COMPANY = "date_sent_to_company";
inline const std::string PRODUCT = "product";
inline const std::string SUB_PRODUCT = "sub_product";
inline const std::string ISSUE = "issue";
inline const std::string SUB_ISSUE = "sub_issue";
inline const std::string COMPLAINT_TEXT = "complaint_text";
inline const std::string COMPANY = "company";
inline const std::string STATE = "state";
inline const std::string ZIP_CODE = "zip_code";
inline const std::string TAGS = "tags";
inline const std::string SUBMITTED_VIA = "submitted_via";
inline const std::string TIMELY_RESPONSE = "timely_response";
inline const std::string COMPANY_RESPONSE_TO_CONSUMER = "company_response_to_consumer";
inline const std::string COMPLAINT_TEXT_LENGTH = "complaint_text_length";
inline const std::string HAS_NARRATIVE = "has_narrative";
inline const std::string MONETARY_RELIEF = "monetary_relief";

}  // namespace columns

// The raw CFPB CSV header names live in a separate namespace because they belong
// to the input side and are distinct from the transformed output columns above.
namespace raw {

// Raw CFPB CSV header for the consumer narrative field the ETL filters on.
inline const std::string SOURCE_NARRATIVE_COLUMN = "Consumer complaint narrative";

}  // namespace raw

// One expected output column: its name paired with the Arrow type it must be
// emitted as.
struct ExpectedColumn {
  std::string name;
  std::shared_ptr<arrow::DataType> type;
};

// Keep the expected schema centralized so the ETL output contract is easy to
// review as the ML feature set changes.
inline std::vector<ExpectedColumn> getExpectedSchema() {
  return {
    {columns::COMPLAINT_ID, arrow::int64()},
    {columns::DATE_RECEIVED, arrow::date32()},
    {columns::DATE_SENT_TO_COMPANY, arrow::date32()},
    {columns::PRODUCT, arrow::utf8()},
    {columns::SUB_PRODUCT, arrow::utf8()},
    {columns::ISSUE, arrow::utf8()},
    {columns::SUB_ISSUE, arrow::utf8()},
    {columns::COMPLAINT_TEXT, arrow::utf8()},
    {columns::COMPANY, arrow::utf8()},
    {columns::STATE, arrow::utf8()},
    {columns::ZIP_CODE, arrow::utf8()},
    {columns::TAGS, arrow::utf8()},
    {columns::SUBMITTED_VIA, arrow::utf8()},
    {columns::TIMELY_RESPONSE, arrow::boolean()},
    {columns::COMPANY_RESPONSE_TO_CONSUMER, arrow::utf8()},
    {columns::COMPLAINT_TEXT_LENGTH, arrow::int32()},
    {columns::HAS_NARRATIVE, arrow::boolean()},
    {columns::MONETARY_RELIEF, arrow::int8()}
  };
}

}  // namespace complaintiq

#endif  // COMPLAINTIQ_SCHEMA_H
