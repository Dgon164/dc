# Cleaning changelog — dirty_sales

60 rows out; every change below; nothing else was touched.

## Changes made (mechanical, receipted)

- STRUCTURE: delimiter detected: ';' (converted to comma)
- STRUCTURE: skipped title row: 'Sales Export FY26 (internal)'
- STRUCTURE: blanked 1 missing-value marker(s) (***, NA, null, ...)
- WHITESPACE: trimmed 1 cell(s)
- DUPLICATE: row 23 removed — exact duplicate of row 22
- DUPLICATE: row 37 removed — exact duplicate of row 36

## Left for your judgment (we flag, we don't guess)

- audit view: 'date' dates converted to day numbers (receipted, view only)
- audit view: 'region' excluded (text — cleaned but not numerically auditable)
- engine audit (0s): verdict NO RULE; 0 error / 0 news / 0 duplicate / 59 unvalidatable / 0 unflagged
- FLAGGED FOR REVIEW: refused (no applicable law — rows counted unvalidatable, never silently passed)
- PASS B (0s): with 1 extreme value(s) held out (receipts below), the law CERTIFIED on the remaining 59 rows — verdict PARTIAL; 0 error / 0 news / 0 duplicate / 0 unvalidatable / 58 unflagged
- FLAGGED FOR REVIEW — OUTLIER-CANDIDATE: row 43: revenue=999,999.99 sits 4,859 MADs from the median 1,344.76 (threshold 8); inconsistent with the law the rest of your data obeys. Kept in the cleaned file; your call.

A flagged cell is evidence, not a typo — overwriting it silently is how other people's cleaning loses real signal. Tell us the intended value and we apply it, receipted.
