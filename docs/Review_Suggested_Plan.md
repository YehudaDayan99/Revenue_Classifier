I reviewed `developer_review_v3.zip` end-to-end (DEV_PROPOSAL, PIPELINE_FLOW, repo code, run_report, and the manual-inspection comparison CSVs).

## Conclusion

**(2) Suggest changes / improvements (do not sign off yet).**
The architecture is broadly correct and *does not* need a major revamp, but there are two **production-blocking reliability gaps** and a few **must-fix engineering details** to make the “product/service-only” descriptions consistently high quality at scale.

---

## What is solid (keep)

* **Two-tier evidence strategy** (table-local → full-filing) is the right backbone.
* **Note Revenue classification** (`note_revenue_sources` vs `note_revenue_recognition`) is the correct move and is already paying off (META/MSFT quality).
* **Post-generation accounting sentence filtering** is directionally correct.
* The dev plan’s Tiering (NVDA → AMZN → GOOGL → AAPL) is the right execution order.

---

## Production blockers to fix before sign-off

### Blocker A — Validation is currently unsafe (accepts obviously wrong extractions)

In `revseg/extraction/validation.py`, the fallback path **accepts** cases where:

* `table_total` exists but the extraction is wildly off (META example).
* `external_total` exists but extraction is >2x the external total (NVDA example).
  This is why `run_report.json` shows “ACCEPTED … no validation reference available” even when there *is* a reference and it’s badly violated.

**Required changes**

1. If `table_total` exists and fails tolerance ⇒ **do not allow fallback acceptance**. Fail and retry next-best table.
2. If `external_total` exists and fails tolerance ⇒ **do not allow fallback acceptance**. Fail and retry next-best table.
3. Add symmetric sanity bounds when external exists:

   * reject if `segment_sum / external_total < 0.85` or `> 1.15` (tune), not just `< 0.1`.

This is non-negotiable for scaling to hundreds of companies: you need hard QA gates that prevent silent bad outputs.

---

### Blocker B — NVDA is not a “description issue”, it’s a layout inference + missing guardrails issue

NVDA shows currency values as “Revenue Line” because `infer_disaggregation_layout()` can pick the wrong `item_col`, and nothing validates it.

**Required changes**
Implement a deterministic `choose_item_col(grid)` and override LLM output when:

* `numeric_ratio(item_col) > threshold` OR
* too many labels match `^\$?\d` / currency patterns OR
* entropy/uniqueness is low relative to another column

Then add a post-extraction check:

* If >50% of extracted “Revenue Line” values are numeric/currency ⇒ **invalidate extraction and rerun layout/table selection**.

---

## “Product/service-only” description quality gaps and fixes

### 1) AAPL “Services” remains incomplete due to heading extraction logic

Your current `_extract_heading_based_definition()` finds the “Services” heading but then stops at the **next heading**, which is “Advertising” — exactly the opposite of what you need.

**Required change**
Implement **parent-heading aggregation**:

* When the parent label heading is found, include subsequent sibling blocks **including subheadings and their paragraphs** until a same-level “peer” heading outside the subtree.
* This is best done by DOM traversal (not regex cutoffs on raw HTML).

### 2) AMZN missing lines are a table-footnote evidence problem

Your table-local context builder looks at **siblings after the table**, but AMZN footnotes are often:

* embedded in `<tfoot>` / nested table structures, or
* preceded by underscore rules, or
* not captured in the “content blocks” heuristic.

**Required change**
Extend table-local evidence extraction to include:

* `<tfoot>` text and any footnote-like rows inside the table,
* following siblings until the next table/major heading (not “N blocks”),
* normalized text via `get_text(" ", strip=True)` (regex over raw HTML is brittle here).

Then: treat footnote definitions as **highest-priority evidence** (before RAG).

### 3) GOOGL “Google Network” is definitional text in Item 7

The current plan suggests query expansion. That can work, but it’s brittle.

**Preferred fix (more generalizable)**
Add an `item7_definitions` subtype during chunking:

* In Item 7, split out bullet/definition lines containing “consists of / includes / revenues generated from…”
* Mark those chunks as preferred **only for definitions**, while keeping general MD&A de-preferred.

This handles Google-style bullet definitions without opening the door to “performance drivers” text.

---


## What I still need from you (to finalize sign-off criteria)

1. **Acceptance criteria**: what is the required pass threshold across a validation suite? (e.g., 95% description coverage + 100% numeric reconciliation on external totals when available?)
2. **Runtime/cost budget per filing**: do you have a hard constraint (e.g., <60s, <$X per 10-K)? This affects whether we do deeper DOM traversal and how many retrieval attempts we allow.
3. **Policy on Item 7**: are you okay including *definition-only* chunks from Item 7 (as recommended), or is Item 7 strictly banned regardless?

If you answer those three, I can give a clean “go/no-go” sign-off rubric and a minimal regression harness spec for CI.
