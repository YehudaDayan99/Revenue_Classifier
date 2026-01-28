Below is a **code-focused diagnosis + fix plan** aimed **only** at improving:

> **`Line Item description (company language) — product/service only`**

…to the quality level we produced for AAPL/AMZN, while staying **generalizable to 100s of issuers**.

---

## What is going wrong in the current repo output (AAPL + AMZN)

### A) AMZN: missing descriptions (NaN) is mainly a **footnote-linking failure**

Your legacy path (`use_rag=False` default in `pipeline.run_pipeline(...)`) relies on `_extract_footnote_for_label()` in `react_agents.py` to extract definitions like “(3) Includes commissions…”.

But `_extract_footnote_for_label()` **only runs if the row label contains a footnote marker** like `"Online stores (1)"`. In your produced CSV1, AMZN row labels appear to have **no `(n)` markers**, likely because the table parser loses superscripts/footnote references in iXBRL HTML (common).

Result:

* Footnote extraction does not trigger → no definitions → LLM fallback search often misses or muddles categories → NaNs / weak text.

### B) AAPL: descriptions drift into MD&A “drivers” and/or accounting language

Even when it finds the right area, the legacy description logic:

* pulls evidence windows that include “increased due to…” performance commentary (MD&A-ish),
* and it does not **explicitly exclude** accounting / revenue recognition text at prompt level.

Result:

* Descriptions become “net sales increased due to Pro models…” (not a product definition)
* Services sometimes includes amortization language (accounting), which you want excluded.

---

## 1) Fixes that *should* be sufficient (incremental, scalable)

### Fix 1 — Make footnote extraction work **even when the label lacks (n)**

**Goal:** reliably attach footnote numbers to table rows for iXBRL HTML.

**Where to implement:**

* In the table extraction stage where you already have access to the table DOM:

  * `pipeline.py` (around where `table_elem` is found for RAG local chunks)
  * or in `extract_table_grid_normalized(...)` / whatever function builds the row labels

**Approach (robust + scalable):**

1. When you parse the selected table element (`<table id=...>`), for each row label cell:

   * Extract visible text (current behavior)
   * Also extract any **superscripts / footnote anchors** inside that cell:

     * `<sup>1</sup>`, or
     * `<a href="#...">1</a>`, or
     * `ix:footnoteReference` patterns depending on filer formatting
2. Store footnote_id on the extracted row, e.g.:

   * `ExtractedRow.footnote_ids: List[str]` (new field), or
   * append it back to the label **only for description extraction** (not for the final “Revenue Line” column), e.g. `Online stores (1)`.

**Why this matters:**

* For AMZN, the authoritative product/service definitions are in those “Includes …” footnotes.
* Once you recover footnote ids, `_extract_footnote_for_label()` starts working and your NaNs largely disappear.

---

### Fix 2 — Post-filter accounting/regulatory sentences (deterministic)

Even with good retrieval, you need a **hard filter** to strip accounting/reporting language.

**Where to implement:**

* After `_extract_footnote_for_label()` returns text (legacy path)
* And after LLM returns description (legacy + RAG path)

**Simple scalable method: sentence-level removal**
Split into sentences and drop any sentence matching a denylist such as:

* revenue recognition / accounting:

  * `performance obligation`, `stand-alone selling price|ssp`, `allocated`, `recognized`, `deferred`, `amortization`, `contract liability`, `unearned`, `ASC`, `GAAP`
* reporting mechanics:

  * `record revenue`, `gross`, `net of`, `consolidated`, `included in`, `reclassification`
* table mechanics:

  * `see note`, `refer to`, `in the table`, `as shown`

For AMZN “Online stores” footnote, this will remove the “record revenue gross” sentence while retaining the product/service content.

This gives you a deterministic guardrail that scales.

---

### Fix 3 — Tighten the prompt: explicitly forbid accounting + forbid “drivers”

In `react_agents.describe_revenue_lines(...)` (legacy) and in `rag/generation.py` (RAG), add explicit constraints:

* **Must describe the product/service offered**, not performance/changes
* **Exclude**: revenue-recognition, accounting policy, “increased due to…”, “net sales decreased due to…”
* Prefer definitions that look like:

  * “X is the Company’s line of …”
  * “Includes …”
  * “Provides … services such as …”

You already have a strong structure; it just needs stricter instruction.

---

### Fix 4 — Evidence selection: de-prioritize MD&A “drivers” and revenue recognition policy

In the legacy approach, evidence is built from:

* `[TABLE CONTEXT]`, `[ITEM1]`, `[ITEM7]`, `[ITEM8]`

For “product/service only”, **ITEM7 is often harmful** (drivers commentary).
Recommendation:

* For description extraction, change priority to:

  1. table footnotes / table-local context (best for disaggregation definitions)
  2. Item 1 Business (best for Apple-style product definitions)
  3. Item 8 notes **only if it’s definitional** (not revenue recognition policy)
  4. Item 7 MD&A: last resort or disable

This alone will reduce the “higher sales of Pro models” type of leakage.

---

## 2) If incremental fixes aren’t sufficient: an alternative approach (more reliable at scale)

If you want a **step-change in description quality** without fragile global searching, implement a **two-source “definition resolver”** that is mostly deterministic and only uses the LLM for summarization.

### Alternative: Definition Resolver (scalable pattern)

For each revenue line:

**Source A — Table-local definitional footnotes (highest precision)**

* Parse the selected table DOM and capture:

  * caption
  * immediate footnote blocks after the table
  * map `(n)` footnotes to revenue lines via recovered superscripts
* Extract the “Includes …” / “consists of …” clause(s)
* Apply accounting sentence filter
* Summarize to 1–2 sentences (LLM optional; often not needed)

**Source B — Item 1 product/service section, heading-based extraction (Apple-style)**
Instead of keyword windows, use structure:

* In Item 1, detect headings / bold headers that equal or closely match the revenue line:

  * “iPhone”, “Mac”, “iPad”, “Wearables…”, “Services”
* Extract the next N paragraphs until the next heading
* Apply accounting filter
* Summarize

**Fallback Source C — RAG**
Only if A and B fail, use RAG retrieval with:

* query that targets “products and services included”
* **exclude** “revenue recognition” from the query (your current `build_rag_query()` literally appends “revenue recognition segment note”, which is exactly what you don’t want)

This approach generalizes because:

* Many issuers define disaggregation lines in **table footnotes**
* Many issuers define product categories in **Item 1 headings**
* RAG becomes a last resort rather than the primary engine

---

## 3) Specific repo changes I would make (surgical list)

1. **Enable RAG by default for description extraction** *only if* you want it:

   * `pipeline.run_pipeline(... use_rag=True)` (currently default False)
   * But don’t rely on it until you fix query + filters.

2. **Recover footnote IDs from the selected table DOM**:

   * Add helper: `extract_row_label_and_footnote_ids(table_elem) -> {row_label: [ids]}`
   * Store this mapping and pass it into the description module.
   * Update `_extract_footnote_for_label()` to accept `(label, footnote_id)` or accept mapping directly.

3. **Add `strip_accounting_sentences(text)` utility**

   * Call it in both:

     * legacy footnote extraction output
     * final description output (legacy + RAG)

4. **Prompt changes**

   * `react_agents.describe_revenue_lines()` system prompt: add “exclude accounting/regulatory text; exclude performance drivers”
   * `rag/generation.py` system prompt: same constraints
   * `rag/generation.build_rag_query()`: remove “revenue recognition segment note”

5. **Reduce ITEM7 usage for descriptions**

   * In legacy evidence gathering, either:

     * remove `item7_section` from `sections_priority`, or
     * only include it if Item1/footnotes returned nothing.

---

## Practical expectation

* With **footnote-id recovery + accounting sentence filter + prompt tightening**, you should get **AMZN to near-perfect coverage** and **AAPL to the quality we generated** (product/service definitions, not drivers).
* The “Definition Resolver” approach is the most stable general solution for 100s of filers because it uses the same two structural sources that most 10-Ks consistently provide.

