# Revenue Table Format Patterns

This reference documents the two major revenue disaggregation table formats found in SEC 10-K filings, with extraction strategies for each.

---

## Table of Contents

1. [AAPL-style (Row-based)](#aapl-style-row-based)
2. [MSFT-style (Header-based)](#msft-style-header-based)
3. [Dimension Detection](#dimension-detection)
4. [Common Edge Cases](#common-edge-cases)
5. [Company-Specific Patterns](#company-specific-patterns)

---

## AAPL-style (Row-based)

### Structure

Revenue items appear as row labels in column 0, with year values in subsequent columns.

```
┌─────────────────────────┬──────────────┬──────────────┐
│ Products and Services   │    2025      │    2024      │
├─────────────────────────┼──────────────┼──────────────┤
│ iPhone                  │   $200,583   │   $205,489   │
│ Mac                     │    $29,984   │    $29,357   │
│ iPad                    │    $26,694   │    $28,300   │
│ Wearables, Home & Acc.  │    $37,005   │    $39,845   │
│ Services                │    $96,169   │    $85,200   │
├─────────────────────────┼──────────────┼──────────────┤
│ Total net sales         │   $390,435   │   $388,191   │
└─────────────────────────┴──────────────┴──────────────┘
```

### Layout Inference

```json
{
  "segment_col": null,
  "item_col": 0,
  "year_cols": {"2025": 1, "2024": 2},
  "header_rows": [0],
  "total_row_regex": "total\\s+net\\s+sales",
  "exclude_row_regex": "",
  "units_multiplier": 1000000,
  "notes": "AAPL-style row-based, values in millions"
}
```

### Extraction Strategy

1. Skip `header_rows`
2. For each remaining row:
   - Label = cell at `item_col`
   - Value = cell at `year_cols[target_year]`
3. Classify row types:
   - Match against `total_row_regex` → `total`
   - Match against subtotal patterns → `subtotal`
   - Otherwise → `item`

### Companies Using This Format

- **AAPL**: Products vs Services breakdown
- **GOOGL**: Revenue by property (Google Search, YouTube, Cloud)
- **NVDA**: Compute vs Graphics segments
- **COST**: Merchandise + Services breakdown
- **XOM**: Upstream/Downstream product categories

---

## MSFT-style (Header-based)

### Structure

Segment names appear as header rows (often bold or spanning). Beneath each segment: metric rows including "Revenue".

```
┌─────────────────────────────────────────────────────────┐
│ Productivity and Business Processes                    │  ← Segment header
├─────────────────────────────────────┬─────────┬────────┤
│ Revenue                             │ $68,424 │$63,364 │  ← Extract
│ Operating income                    │ $34,189 │$31,651 │  ← Skip
├─────────────────────────────────────────────────────────┤
│ Intelligent Cloud                                       │  ← Segment header
├─────────────────────────────────────┬─────────┬────────┤
│ Revenue                             │ $97,692 │$87,907 │  ← Extract
│ Operating income                    │ $48,714 │$42,515 │  ← Skip
├─────────────────────────────────────────────────────────┤
│ More Personal Computing                                 │  ← Segment header
├─────────────────────────────────────┬─────────┬────────┤
│ Revenue                             │ $59,672 │$54,734 │  ← Extract
│ Operating income                    │ $18,584 │$16,450 │  ← Skip
└─────────────────────────────────────┴─────────┴────────┘
```

### Layout Inference

```json
{
  "segment_col": null,
  "item_col": 0,
  "year_cols": {"2024": 1, "2023": 2},
  "header_rows": [0],
  "total_row_regex": "^\\s*total\\s*$",
  "exclude_row_regex": "operating\\s+(income|expense)|cost\\s+of",
  "units_multiplier": 1000000,
  "notes": "MSFT-style header-based segments"
}
```

### Extraction Strategy

**Detection**: Scan for pattern where:
1. A row's first cell matches a known segment name
2. The next row's first cell is "Revenue" (case-insensitive)

**Algorithm**:
```
current_segment = null
for each row:
    first_cell = row[0].lower().strip()
    
    if first_cell in expected_segments:
        current_segment = first_cell
        continue  # This is a header row
    
    if first_cell == "revenue" and current_segment:
        extract value for current_segment
        current_segment = null  # Reset after extraction
```

### Companies Using This Format

- **MSFT**: Three operating segments (Productivity, Cloud, Personal Computing)
- **AMZN**: North America, International, AWS (segment results format)
- **META**: Family of Apps, Reality Labs
- **V**: Service revenues, Data processing revenues, etc.

---

## Dimension Detection

Tables can disclose revenue along different dimensions. Detect from captions, headings, and row labels.

### Product/Service Dimension (Most Granular)

**Indicators in caption/heading:**
- "groups of similar products and services"
- "disaggregation of revenue"
- "revenue by product category"
- "net sales by type"

**Example row labels:**
- iPhone, Mac, iPad, Services (AAPL)
- Google Search, YouTube Ads, Google Cloud (GOOGL)
- Online stores, Third-party seller services, AWS (AMZN)

### Segment Dimension

**Indicators:**
- "reportable segments"
- "operating segments"
- "segment results"
- "revenue by segment"

**Example row labels:**
- Productivity and Business Processes, Intelligent Cloud, More Personal Computing (MSFT)
- Family of Apps, Reality Labs (META)
- North America, International, AWS (AMZN segment view)

### Geography Dimension (Usually Skip)

**Indicators:**
- "by geographic area"
- "revenue by region"
- "Americas, EMEA, Asia Pacific"

Geography tables are typically skipped in favor of product/segment tables unless explicitly requested.

### Priority Order

When multiple dimension tables exist, prefer:
1. `product_service` — Most granular breakdown
2. `segment` — Business segment level
3. `geography` — Skip unless specifically needed

---

## Common Edge Cases

### Merged Cells / Rowspan

Some tables use rowspan for segment names, leaving subsequent rows blank in column 0.

**Solution**: Fill-down — propagate last non-empty segment value to subsequent rows.

```
│ North America │ Online stores    │ $98,876 │
│               │ Physical stores  │ $19,977 │  ← Segment = "North America"
│               │ Third-party      │ $64,389 │  ← Segment = "North America"
│ International │ Online stores    │ $32,106 │  ← Segment = "International"
```

### Currency Symbol in Separate Column

Some tables put "$" in its own column, shifting value indices.

```
│ iPhone │ $ │ 200,583 │ $ │ 205,489 │
```

**Solution**: If year column yields `$`, try `year_col + 1`.

### Parentheses for Negatives

Financial tables represent negatives as `(1,234)` not `-1,234`.

**Solution**: `_parse_money_to_int` handles both formats:
```python
if "(" in val and ")" in val:
    val = val.replace("(", "-").replace(")", "")
```

### Footnote Markers

Row labels often include footnotes: `"iPhone (1)"`, `"Services (2)(3)"`

**Solution**: Strip with regex before matching:
```python
label = re.sub(r'\s*\(\d+\)\s*$', '', label).strip()
```

### Units in Header vs. Caption

Units may appear in:
- Table caption: "Revenue (in millions)"
- Header row cell: "(in millions)"
- Standalone row: "Dollars in millions except per share data"

**Solution**: Check all three locations. Common multipliers:
- "thousands" → 1,000
- "millions" → 1,000,000
- "billions" → 1,000,000,000

---

## Company-Specific Patterns

### NVDA (Compute & Graphics)

Two segments with product-level detail. Watch for:
- "Data Center" as a product line within Compute
- "GeForce" within Graphics
- Hedging adjustments as reconciling items

### GOOGL (Hedging Gains)

Includes "Hedging gains (losses)" as a separate line item below segments. This is an **adjustment** row, not a revenue line.

```json
{"item": "Hedging gains (losses)", "row_type": "adjustment"}
```

### AMZN (Product vs. Service)

Two overlapping views:
- **Product sales** vs **Service sales** (dimension: product_service)
- **Segment view**: North America, International, AWS (dimension: segment)

Prefer the product/service disaggregation for granularity.

### META (Single Segment Reality)

Despite two "segments" (Family of Apps, Reality Labs), META's main revenue table shows:
- Advertising
- Other revenue

Extract at this granular level, not segment totals.

### JPM / Banks (Special Case)

Banks have "Net interest income" which doesn't fit product/service model. Often requires custom handling or explicit skip.

### BRK-B (Conglomerate)

Berkshire reports subsidiaries as segments with wildly different businesses. Revenue aggregation may not be meaningful across insurance, railroad, utilities.

---

## Detection Functions

### `_detect_segment_header_mode(grid, expected_segments)`

Returns `True` if table appears to use MSFT-style header-based format.

**Logic**:
1. Scan first 50 rows
2. If a row's first cell matches a segment name AND next row's first cell is "Revenue" → return `True`

### `detect_dimension(caption, heading, row_labels, ticker)`

Classifies table dimension based on text patterns.

**Returns**: `"product_service"`, `"segment"`, `"geography"`, or `"unknown"`

**Priority**: Product/service patterns checked first (most specific), then geography, then segment, then fallback to unknown.

---

## Validation Patterns

### Subtotal Detection

Labels that indicate subtotals (not leaf items):
- Exact segment names when granular items also present
- "Total <segment_name>"
- "Subtotal"
- Known segment-level labels for specific tickers

### Adjustment Detection

Labels that indicate reconciling adjustments:
- "Hedging gains (losses)"
- "Corporate"
- "Eliminations"
- "Intercompany"
- "Unallocated"

### Skip Patterns

Labels to exclude entirely:
- "Deferred revenue"
- "Contract liability"
- "Unearned"
- Cost/expense lines
- Percentage rows

---

## Example: Full Extraction Flow

**Input**: NVDA 10-K revenue table grid

**Step 1**: Detect format
```
_detect_segment_header_mode → False (AAPL-style)
detect_dimension → "segment" (based on "Compute" and "Graphics" labels)
```

**Step 2**: Infer layout
```json
{
  "segment_col": null,
  "item_col": 0,
  "year_cols": {"2025": 1, "2024": 2},
  "header_rows": [0, 1],
  "total_row_regex": "total\\s+revenue",
  "units_multiplier": 1000000
}
```

**Step 3**: Extract values
```
Row 2: "Data Center" → segment=Compute, item=Data Center, value=115205
Row 3: "Compute" → row_type=subtotal (skip if granular exists)
Row 4: "Graphics" → segment=Graphics, item=Graphics, value=14543
Row 5: "Hedging" → row_type=adjustment, value=-165
Row 6: "Total revenue" → row_type=total, value=130497
```

**Step 4**: Validate
```
segment_sum = 115205 + 14543 = 129748
adjustment_sum = -165
computed_total = 129583
table_total = 130497
delta = 0.7% → OK (within 2% tolerance)
```

**Output**: Validated extraction with 2 items, 1 adjustment
