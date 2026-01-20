Below is a developer-facing review document of the **current outputs** you attached:

* `csv1_segment_revenue.csv` (revenue line items)
* `csv2_segment_descriptions.csv` (segment descriptions)
* `csv3_segment_items.csv` (segment item lists + descriptions)

I validated the outputs against the three attached 10-K HTMLs:

* `msft-20250630.htm`
* `goog-20241231.htm`
* `aapl-20250927.htm`

I also inspected the execution logic in the zipped codebase you provided (`revseg.zip`) after unpacking it locally (notably `revseg/react_agents.py`, `revseg/extraction/core.py`, and `revseg/extraction/validation.py`).

---

# Developer Review: Revenue + Segment Description Pipeline (MSFT / GOOGL / AAPL)

## 0) Executive summary (what works vs what does not)

### Works (high confidence)

1. **Revenue table selection + numeric extraction** is working for these three filings.
2. **Latest-year revenues** for extracted line items match the values in the filings.
3. **Percent-of-total reconciliation** is enforced successfully (each company sums to 100% in `csv1_segment_revenue.csv`).

### Does not work (material issues)

1. **Segment descriptions (csv2)** are not consistently grounded in the correct 10-K sections, especially for **Alphabet**.
2. **Item lists (csv3)** are frequently **hallucinated, outdated, duplicated, or mis-assigned to segments**, especially for **Alphabet** and partially for **Microsoft**.
3. The pipeline currently mixes “reconciliation adjustments” (e.g., hedging) into product/service outputs (policy mismatch with the stated objective: “products and services with explicit revenue”).

---

## 1) Q1 — Are all business segments identified?

### Microsoft (MSFT)

* Expected reportable segments: **Productivity and Business Processes**, **Intelligent Cloud**, **More Personal Computing**.
* Output status:

  * `csv1_segment_revenue.csv`: **Yes**, the three segments are present (as the “Segment” field used to group revenue items).
  * `csv2_segment_descriptions.csv` / `csv3_segment_items.csv`: includes an additional segment **“Other”**.
* Assessment:

  * **Core segments are identified.**
  * **“Other”** should not be treated as a business segment unless explicitly required; it appears to be a residual revenue line, not a reportable segment.

### Alphabet (GOOGL)

* Expected reportable segments: **Google Services**, **Google Cloud**, **Other Bets**.
* Output status:

  * `csv1`: Yes.
  * `csv2`: Yes (3 segments).
  * `csv3`: Yes (3 segments) — but item assignments are problematic (see Q2/Q4).

### Apple (AAPL)

* Apple is effectively one reportable segment operationally, but the filing provides explicit **net sales by product category**.
* Output status:

  * Pipeline treats product categories as “segments”: **iPhone, Mac, iPad, Wearables/Home/Accessories, Services**.
* Assessment:

  * This is acceptable **if the design definition** for Apple is “revenue product categories.”
  * If the design definition is “reportable operating segments,” then Apple handling is not aligned.

**Developer action:** formalize the definition of “Segment” per ticker class:

* MSFT/GOOGL: reportable segments.
* AAPL: product categories (explicit net sales categories).
* Avoid introducing “Other” as a segment unless it is explicitly defined as such in the filing.

---

## 2) Q2 — Are all sub-segment services/products identified correctly?

### Revenue sub-items in `csv1_segment_revenue.csv` (GOOD)

* MSFT: the expected explicit revenue lines (e.g., Server products and cloud services, Microsoft 365 Commercial, Gaming, LinkedIn, etc.) are captured and mapped into the three MSFT segments. Numeric values align with the filing.
* GOOGL: revenue lines such as Search & other, YouTube ads, Network, Subscriptions/platforms/devices, Cloud, Other Bets are captured with correct numbers.
* AAPL: product category net sales captured with correct numbers.

### “Items under a segment” in `csv3_segment_items.csv` (NOT RELIABLE)

This is the major gap.

#### Alphabet

* **Mis-assignment:** `Google Cloud Platform` and `Google Workspace` appear under **Google Services** as well as Google Cloud.
* **Hallucinations / outdated entities:** examples include items that are unlikely to be present in the current 10-K (e.g., “Loon”).
* Net: **segment→item mapping is not trustworthy** for GOOGL.

#### Microsoft

* Many items look plausible, but there are issues:

  * **Duplicate entries** (LinkedIn sub-items repeated).
  * Several items did not match exact text in the filing via simple string presence checks (may be phrasing variance, but also may be model expansion).
  * “Other” segment items are essentially duplicates of other segments, which is conceptually wrong.

#### Apple

* Generally plausible product lists, but a few items did not match exact-string presence (likely formatting/encoding issues like “TV+” in HTML), and some entries may be expansions beyond the excerpted source text.

**Developer action:** treat `csv3` as “LLM-generated candidates” unless and until grounding + validation is enforced (see Section 6).

---

## 3) Q3 — Are the latest revenue numbers captured correctly?

### Result: YES (for extracted rows)

I cross-checked the extracted latest-year revenues for the chosen disaggregation tables:

* MSFT uses a table that includes values like **98435** (in millions) for Server products and cloud services, which your output converts to **$98,435,000,000**.
* AAPL and GOOGL behave similarly; units are being applied correctly.

### Caveat: “adjustments” included

* GOOGL output includes **“Hedging gains (losses)”** as a revenue line in `csv1` with `Row type = adjustment`.
* This improves reconciliation to 100%, but it violates the business rule you stated (“include only products and services”).

**Developer action:** separate:

* **Product/service revenue lines** (reportable disaggregation categories)
* **Reconciliation adjustments** (hedging/corporate/rounding)
  and decide whether adjustments are allowed in the deliverable table or only used internally for validation.

---

## 4) Q4 — Did the process correctly identify all items mentioned in the 10-K?

### Revenue items (csv1): YES (for the target disaggregation tables)

The extracted explicit revenue line items appear complete for the selected disaggregation tables.

### Segment items (csv3): NO (in its current form)

A simple grounding test (case-insensitive substring presence of each `Business item` in the source 10-K HTML) shows:

* **Alphabet:** ~38% of items matched; ~62% did not.
* **Microsoft:** ~79% matched; ~21% did not (plus duplicates).
* **Apple:** ~89% matched; ~11% did not.

This is consistent with the current prompting strategy in `expand_key_items_per_segment()` which asks the model to “expand into 5–10 items” without enforcing that the items **must** be extracted from the provided text snippet.

**Developer action:** convert `csv3` generation from “expansion” to “extraction with evidence,” with strict post-validation (Section 6).

---

## 5) Q5 — Did the process correctly describe the items? Are descriptions accurate and rich enough for downstream classification?

### Richness

* Descriptions are reasonably long (often 30–60 words long-form). That is good for downstream classification.

### Accuracy (mixed)

* When the item itself is hallucinated or mis-assigned (notably GOOGL), the description becomes **unreliable** regardless of richness.
* Even when the item is real, the description is often **generic** and not anchored to filing language (e.g., “focuses on innovation, competitive markets…”). That is not ideal for building a robust classifier over filing tables.

### Segment descriptions (csv2) — Alphabet is problematic

Example pattern:

* “Google Services encompasses … devices … wearables … cloud services” is not the canonical segment framing in the segment note; it looks like the LLM is inferring rather than citing.

**Developer action:** rewrite description generation so it is:

1. **Evidence-first:** derived from the actual segment note text (or other designated section).
2. **Structured:** include (a) what it is, (b) how it monetizes, (c) representative products/services, (d) synonyms/aliases used in tables.
3. **Quoted anchors optional but recommended:** store 1–3 short supporting spans for each segment/item.

---

## 6) Root causes in the codebase and concrete fixes

### Root cause A — The snippet retrieval for segment descriptions is weak (esp. GOOGL)

In `summarize_segment_descriptions()` (in `revseg/react_agents.py`), snippets are built by searching for segment names in broad text windows, with heuristics around “segment information” / “note 18”.

* This is brittle for Alphabet because segment descriptions may not align with those heuristics or may appear multiple times with different contexts.

**Fix A1 — Anchor to the segment footnote/table**

* Identify the **segment note section** more deterministically (e.g., find the “Segment information” note header, then bound the section until the next note).
* Extract per-segment paragraphs by regex boundaries:

  * Start at “Google Services” paragraph heading, end at next segment heading.
* Feed those bounded paragraphs to the LLM.

**Fix A2 — Use the revenue disaggregation labels as “must include” keywords**

* Since you already extracted the revenue items (Search & other, YouTube ads, etc.), pass those labels into the segment description prompt and require the model to map them into the correct segment description.

---

### Root cause B — `expand_key_items_per_segment()` is generative, not extractive

Current prompt: “expand into 5–10 key product/service items” with no grounding requirement.

**Fix B1 — Convert to extract-only + evidence**
Change the contract to:

* The model must output only items that appear verbatim in the snippet.
* Each item must include an `evidence_span` string copied from the snippet (10–25 words max).
* Post-validate: reject any item where `evidence_span` is not found in the snippet.

**Fix B2 — Deterministic extraction fallback**
Before the LLM:

* run a deterministic “candidate noun-phrase/product” extractor on the snippet:

  * capitalized sequences, trademark-like tokens, known product dictionaries, etc.
* Then ask the LLM to *filter and describe* those candidates, not invent new ones.

**Fix B3 — De-duplication and segment-consistency**

* De-dup by normalized key (`lower`, strip punctuation).
* Enforce “segment must match input segment name” (you already do this), but also enforce:

  * no cross-segment duplicates unless explicitly allowed and justified.

---

### Root cause C — Adjustments included in product/service output (policy mismatch)

In unified extraction + validation (`revseg/extraction/core.py` + `validation.py`), adjustments are treated as part of “computed total” to pass reconciliation.

**Fix C1 — Dual-output schema**
Emit:

* `revenue_items` (product/service categories)
* `revenue_adjustments` (hedging/corporate/rounding)
* `table_total`

Then:

* Use `items + adjustments` to validate,
* But export only `items` unless user explicitly requests adjustments.

---

## 7) Proposed acceptance criteria (tests the developer can implement)

### Revenue extraction (csv1)

For each ticker:

1. The chosen table contains the latest year column and a Total row.
2. All extracted revenue items have explicit numeric values for the latest year.
3. Sum of exported items equals 100% ± 0.01 after rounding.
4. Adjustments (if any) are **not** included in the exported “products/services” table unless explicitly requested.

### Segment descriptions (csv2)

1. Each segment description must include at least N “filing-native” keywords (exact-string matches from the snippet).
2. Each segment row includes a `source_span` (supporting evidence text).

### Segment item lists (csv3)

1. Every item must be found in the filing text (exact match or validated fuzzy match).
2. Every item must include an evidence span found in the filing.
3. No duplicate items per segment after normalization.
4. Alphabet: zero “Google Cloud Platform / Workspace” under Google Services unless explicitly supported by the filing text.

---

## 8) Priority worklist for the developer

1. **Implement evidence-based extraction for `csv3`** (most important).
2. **Improve snippet bounding for Alphabet segment descriptions** (second most important).
3. **Split revenue adjustments from revenue items** (prevent hedging from appearing as a “product/service”).
4. Add a `--strict` mode that fails the run if grounding checks fail (instead of silently outputting hallucinations).

---
