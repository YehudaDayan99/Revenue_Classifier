---

## 0) Current state (what works, what fails)

### Works (mostly)

* **AMZN**: the new **DOM footnote ID recovery** + footnote extraction path is the right approach and is already integrated in `pipeline.py` → `react_agents.describe_revenue_lines(... footnote_id_map=...)`. This is the highest-precision source for “what it is” definitions.

### Fails / weak

* **AAPL – Services = NaN** in current output (`csv1_segment_revenue.csv` inside the zip). Root cause: keyword-window search for “Services” is too ambiguous and often lands in non-definitional contexts; **heading-based extraction is needed** (as already proposed in `DESCRIPTION_EXTRACTION_PROPOSAL.md` but not implemented in `react_agents.py` yet).
* **META – Other revenue = NaN**, and existing “Advertising” / “Reality Labs” descriptions are **generic** (low “company language” richness). Root cause: Note 2 contains **definition + adjacent revenue recognition mechanics**; section-level routing isn’t enough—needs **chunk-level classification / filtering**.
* **NVDA – all lines = NaN**. Root cause: product/service descriptions are mostly in **Item 1 narrative**; legacy keyword windows don’t reliably retrieve the right definitional paragraphs. **RAG (or a structured Item 1 resolver) is required**.

---

## 1) Recommendation: implement a “Definition Resolver” (deterministic-first, RAG fallback)

To scale to 100s of issuers, don’t rely on one retrieval strategy. Use a **ranked resolver**:

### Source A (highest precision): Table-local footnotes / table-adjacent definitions

* Use existing `extract_footnote_ids_from_table()` (already in `react_agents.py`) + `_extract_footnotes_from_text()`.
* If footnote definition exists → **return it** (after stripping accounting/driver sentences).

**This is the backbone for AMZN-like filers.**

### Source B (high precision): Item 1 heading-bounded extraction

* For labels that are real headings (AAPL “Services”, “iPhone”, etc.):

  * Find `<b>/<strong>/h1/h2/h3` nodes matching the label (fuzzy match allowed).
  * Extract until next peer heading.
  * Summarize to 1–2 sentences.

**This is required to fix AAPL Services and improves many “Product” narratives across issuers.**

### Source C (high recall): RAG retrieval with strict evidence + anti-accounting controls

* Only used if A and B fail.
* Primary for NVDA-style narratives and META’s Note 2 definitional fragments.

This “A→B→C” design is both **more accurate** and **cheaper** than running RAG everywhere.

---

## 2) Concrete code changes (priority-ordered)

### P0 — AAPL Services fix (must-do)

**Add heading-based extractor** (as in your proposal) and invoke it *before* keyword windows.

**Files**

* `react_agents.py`

  * Implement `_extract_heading_based_definition()` (proposed in `DESCRIPTION_EXTRACTION_PROPOSAL.md`)
  * In `describe_revenue_lines()` for each line needing LLM:

    1. try heading-based extraction inside `item1_section`
    2. if found and length threshold met → run `strip_accounting_sentences()` → accept
    3. else proceed to existing section-aware keyword windows

**Why this matters**

* “Services” is ambiguous in plain search. Heading-bounded extraction is structurally stable for Apple.

---

### P0 — RAG query + prompt are currently *pulling the wrong content* (must-do for META/NVDA)

Right now `build_rag_query()` **adds** `"revenue recognition segment note"` which is exactly what you’re trying to exclude.

**Files**

* `rag/generation.py`

  * In `build_rag_query()`: remove `"revenue recognition segment note"`.
  * Replace with definitional intent terms, e.g.:

    * `"consists of includes comprises generated from provides"` (these are highly predictive of definitional sentences)

**Also update RAG system prompt** in `generate_description_with_evidence()`:

* Add explicit exclusion rules:

  * exclude revenue recognition mechanics (performance obligations, SSP, allocation, control transfer, principal/agent, ASC/GAAP)
  * exclude MD&A performance drivers (“increased due to…”, “driven by…”, YoY)
* Add “return empty if only accounting/performance text is present”.

---

### P1 — Note 2 “definition vs recognition” split (key to META “Other revenue”)

You already wrote the right approach in `DESCRIPTION_EXTRACTION_PROPOSAL.md`, but it is **not implemented** in `rag/chunking.py`.

**Files**

* `rag/chunking.py`

  * Add `classify_note_revenue_chunk()` and re-label `section` from:

    * `note_revenue` → `note_revenue_sources` or `note_revenue_recognition`
* `rag/index.py`

  * Add `BLOCKED_SECTIONS = { "note_revenue_recognition", "item1a", ... }`
  * Update preferred sections to include `note_revenue_sources` and remove `item7` from preferred (MD&A is a frequent contamination source for “drivers”).

This is the single most important fix for META-quality.

---

### P1 — Apply the accounting/performance sentence filter everywhere

You already have `strip_accounting_sentences()` in `react_agents.py`, but **RAG output doesn’t pass through it**.

**Files**

* `rag/generation.py`

  * After validated output, run:

    * `description = strip_accounting_sentences(description)`
  * If the filter wipes the content (too short), return empty and flag it in QA artifact.

Also filter candidate product extraction:

* In `extract_candidate_products()`, drop accounting tokens (`SSP`, `ASC`, `GAAP`, `IFRS`, `allocation`, etc.) as you proposed.

---

### P2 — Return “Where found in 10-K” as first-class metadata (needed for auditing + iterative improvement)

Right now RAG produces `evidence_chunk_ids` and sections, but CSV1 doesn’t carry “where found”.

**Files**

* `react_agents.py`: in legacy path, for each accepted description, also return a `source` field such as:

  * `Item 8 – Disaggregated net sales footnote (3)`
  * `Item 1 – Business – Services section`
* `rag/generation.py`: build `where_found` from the top evidence chunks (section names + headings + chunk_id)
* `pipeline.py`: store this into an artifact JSON and (optionally) into an additional CSV column.

This is crucial for debugging at scale and for building an evaluation set.

---

## 3) Are the proposed fixes sufficient for META + NVDA?

### META

* **Yes, if you implement P0 + P1.**

  * The required definition exists in Note 2 but is adjacent to accounting; chunk classification + blocked sections + prompt rules will reliably isolate definitional text.

### NVDA

* **Incremental legacy fixes alone are not sufficient.**

  * NVDA’s definitional content is mostly Item 1 narrative; you need **RAG (fixed query/prompt + prefer Item 1)** or a dedicated Item 1 structure resolver.
  * With the proposed RAG fixes + Item 1 prioritization, you should reach strong coverage for 5/6 lines; “OEM and Other” may remain legitimately undefined.

---

## 4) Golden targets for META + NVDA (use as regression fixtures)

Use the “ideal output” already prepared in `manual_inspections/Meta_NVDA_Description_output.csv` as **golden expected behavior**:

### META (3 lines)

* Advertising: marketers advertising on Meta apps; priced by impressions/actions.
* Other revenue: WhatsApp Business Platform, Meta Verified, Payments fees, etc.
* Reality Labs: consumer hardware (Meta Quest, Ray-Ban Meta AI glasses) + related software/content.

### NVDA (6 lines)

* Compute: data-center-scale accelerated computing platform; GPUs/CPUs/DPUs + systems/software.
* Networking: InfiniBand/Ethernet platforms; adapters/cables/DPUs/switches + software stack.
* Gaming: GeForce GPUs + GeForce NOW + platform solutions.
* ProViz: RTX/Quadro, vGPU software, Omniverse Enterprise.
* Automotive: DRIVE hardware/software platform for AV/EV + training/simulation stack.
* OEM and Other: potentially no explicit definition → allow empty.

These fixtures should be turned into automated tests (coverage + “no accounting keywords” constraints).

---

## 5) Minimal execution checklist (developer-ready)

1. **Implement heading extraction** for AAPL Services in `react_agents.py` and wire into `describe_revenue_lines()`.
2. **Fix RAG query + prompt** in `rag/generation.py` (remove “revenue recognition…”, add definitional verbs + exclusion rules).
3. **Implement Note 2 chunk classification** in `rag/chunking.py`; **block recognition chunks** in `rag/index.py`.
4. **Run accounting/performance filter post-generation** in both legacy and RAG paths.
5. **Add “where found” metadata** to outputs (artifact + optional CSV column).
6. **Regression harness**: run AAPL/AMZN/META/NVDA and assert:

   * coverage ≥ target
   * descriptions contain definitional verbs (“includes/consists/comprises/provides”) more often than performance verbs
   * no banned accounting phrases.

---

If you want one guiding principle for the developer: **make the system “extractive-first”** (identify definitional sentences deterministically, then summarize), and treat the LLM as a *compressor*, not a discoverer. This is what makes it generalizable and scalable.
