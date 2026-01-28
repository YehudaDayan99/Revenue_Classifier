I’ve reviewed `developer_review_v3.zip` (pipeline flow, repo files, current-vs-manual comparisons, and the dev next-step plan in `docs/DEV_PROPOSAL.md`). Overall direction is sound and I’m comfortable **signing off on the approach**, with a few **required changes** to make it production-grade and scalable.

## Sign-off decision

**Sign-off: conditional approval** — proceed with the proposed Tier 0→3 execution order, **but incorporate the changes below** (they’re small but materially reduce tail risk across 100s of issuers).

---

## Required changes to the developer plan

### 1) NVDA fix: don’t just “validate item_col”, add a deterministic *re-selector*

Your Tier 0 proposes rejecting `label_col` if “mostly numeric” and falling back to column 0. That’s good, but for scale you want a deterministic re-selection function:

**Implement `choose_item_col(grid)`** that ranks columns by:

* `numeric_ratio` (lower is better)
* `alpha_ratio` / “contains letters” ratio (higher is better)
* uniqueness of tokens (labels tend to be diverse)
* mean string length excluding punctuation/currency (labels longer than “$”/numbers)

Then:

* Compare LLM-proposed `item_col` vs heuristic-best column.
* If LLM-proposed fails thresholds, **override** with heuristic-best.
* Keep LLM only for (segment_col, year_cols, header_rows, total_row_regex).

This avoids a class of failures where even a stronger LLM occasionally picks a year/value column.

### 2) AMZN footnotes: current failure is likely *HTML tokenization*, not “format”

Your plan assumes parsing `(N) Includes...` format is the issue. In practice, AMZN footnote markers/parentheses are often split across tags, so regex on raw HTML misses them.

**Change**: extract table footnotes by DOM, then regex on **`soup.get_text(" ", strip=True)`** (normalized text), not raw HTML.

Minimal implementation:

* Identify the accepted `<table>` node.
* Collect the text from the table **and** immediate sibling nodes that hold footnotes (common patterns: following `<div>`, `<p>`, or “_____” separator areas).
* Run `_extract_footnotes_from_text(normalized_text)` on that.
* Feed extracted footnote definitions into description generation as **highest-priority evidence**.

This should close the AMZN “Subscription services” (5) and “Other” (6) gap reliably.

### 3) AAPL Services: current heading extractor truncates at first subheading

Your Tier 3 notes this; the root is `_extract_heading_based_definition()` stopping at the next bold/heading tag, which for Apple is typically the first subheading (“Advertising”), so the parent “Services” gets only a short preamble.

**Change**: for parent headings, capture child subheadings.
Practical rule:

* When label match is found (e.g., “Services”), continue collecting until a **same-level or higher-level** heading that is *not* one of the known child headings OR until a max char cap.
* For Apple, accept subheadings where the heading text is in a short allowlist pattern (Title Case words, < 5 tokens) and aggregate the following paragraph blocks.

### 4) GOOGL “Google Network”: don’t rely only on query expansion if Item 7 is de-preferred

In the current code, `item7` (MD&A) is not preferred. That’s reasonable for avoiding “drivers” contamination, but Google’s definitions are **often exactly in Item 7 bullet definitions**.

Two options; I recommend (b):

**(a)** Extend query expansion (as proposed) AND ensure retrieval can still return item7 chunks.
**(b) Better:** create an `item7_definitions` pseudo-section at chunking time:

* In Item 7, split out bullet/definition-style text containing “consists of / includes / revenues generated from…”
* Label those chunks `item7_definitions`
* Add `item7_definitions` to `PREFERRED_SECTIONS` (but keep general item7 unpreferred)

This keeps MD&A “drivers” mostly out, while allowing definitional bullets in.

### 5) Add a hard QA gate for “obviously wrong label columns”

Your `run_report.json` shows NVDA “ACCEPTED” even though the extracted “Revenue Lines” are dollar amounts and segment sums explode. That’s a reliability issue for scale.

Add a **post-extraction QA gate**:

* If >50% of revenue-line labels match currency/number patterns → fail the table/layout and re-run with fallback selection (or stop with explicit error).
* If SEC external total is available and mismatch > tolerance → fail, don’t “ACCEPTED”.

This prevents silent bad outputs.

### 6) Provenance should be first-class (for developer QA + client trust)

Even if CSV doesn’t include it yet, produce a parallel artifact (JSON) per ticker with:

* `revenue_line`
* `description`
* `source_section` (item1 / item8 / note_revenue_sources / table_footnote / etc.)
* `evidence_snippet` (short)
* `table_id` and optional footnote id

You already have most of this in the RAG path; wire it through consistently.

---

## Practical execution plan (developer-ready)

**Phase 1 (Blocker): NVDA**

1. Implement heuristic `choose_item_col()` and override logic.
2. Add “label column sanity gate” (numeric-heavy labels fail fast).
3. Re-run NVDA only; verify labels are: Compute / Networking / Gaming / ProViz / Automotive / OEM & Other.

**Phase 2 (Coverage): AMZN + GOOGL**
4) DOM-based table-footnote extraction using normalized text (`get_text`).
5) Feed extracted footnotes as highest-priority evidence (pre-RAG).
6) Implement `item7_definitions` or, minimum, query expansion + allow item7 retrieval for GOOGL Network.

**Phase 3 (Quality): AAPL Services**
7) Enhance heading extraction to aggregate subheadings under parent headings.

**Phase 4 (Regression harness)**
8) Automated checks vs manual-inspection CSVs:

* coverage per ticker
* banned accounting phrases absent (or below a low threshold)
* definitional verbs present for non-empty descriptions
* provenance exists for every non-empty description

---

## Question: would switching to GPT-5.2 fix everything?

**No.** Upgrading the LLM can reduce *some* errors (e.g., layout inference choices), but the dominant failures in this pack are **engineering/evidence issues**:

* NVDA: wrong `item_col` is a *layout inference + missing validation* problem → needs deterministic guards.
* AMZN: missing descriptions are *footnote extraction from HTML structure* → better model won’t help if the evidence isn’t surfaced correctly.
* AAPL Services: truncation is a *heading parser logic* issue.
* GOOGL Network: retrieval bias + section preference issue.

### Recommendation on models

* Your current pairing (`gpt-4.1-mini` fast + `gpt-4.1` quality) is **sufficient** once the pipeline is fixed.
* If you want extra robustness, use GPT-5.2 **only as a fallback** when QA gates fail (e.g., label column sanity failure, missing descriptions after resolver). This keeps cost predictable while improving tail performance.

If you implement the changes above, I’d expect the existing models to meet your target quality across hundreds of companies with far less variance than “just upgrade the model.”
