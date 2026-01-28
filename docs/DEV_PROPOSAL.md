# CSV1 Development Proposal — Post-P0/P1 Assessment

**Date**: January 28, 2026  
**Status**: Analysis and proposed fixes for production readiness

---

## Executive Summary

After implementing P0 (heading-based extraction, RAG query refinement) and P1 (Note 2 classification, definitional boost), the CSV1 pipeline achieves:

| Ticker | Coverage | Status |
|--------|----------|--------|
| AAPL   | 100%     | ⚠️ Services description incomplete |
| AMZN   | 71.4%    | ❌ 2 lines missing descriptions |
| GOOGL  | 83.3%    | ⚠️ 1 line missing description |
| META   | 100%     | ✅ All descriptions complete |
| MSFT   | 100%     | ✅ All descriptions complete |
| NVDA   | 0%       | ❌ **TABLE EXTRACTION BUG** |

**Overall**: 4/6 tickers functional, 2 require fixes before production.

---

## 1. Detailed Gap Analysis

### 1.1 NVDA — Critical: Table Extraction Failure

**Symptom**: Revenue Line shows dollar amounts ("$116,193") instead of product names ("Compute").

**Root Cause**: The layout inference agent is identifying the wrong column as the label column. NVDA's table has a structure where:
- Column 0: Product/market names ("Compute", "Networking", "Gaming", etc.)
- Column 1+: Dollar values for each year

The LLM is returning a year column as the label column.

**Evidence** (from current output):
```
NVIDIA CORP,NVDA,2025,"$116,193","$116,193",...
NVIDIA CORP,NVDA,2025,"$82,875","$82,875",...
```

**Fix Required**: 
1. Add validation in `infer_disaggregation_layout()` to reject label_col if most cells are numeric
2. Add heuristic: label_col should have lowest numeric_ratio
3. Add fallback: if detected label_col fails, try column 0

**Effort**: Low (1-2 hours)  
**Impact**: Fixes NVDA completely (6 lines)

---

### 1.2 AMZN — Missing Footnote Descriptions

**Lines Missing**:
- `Subscription services` — Manual: "Amazon Prime memberships, digital video, audiobooks, digital music, e-books, and other non-AWS subscriptions"
- `Other` — Manual: "healthcare services, licensing/distribution of video content, shipping services, co-branded credit card agreements"

**Root Cause**: These descriptions are in **footnote (5)** and **footnote (6)** of the disaggregation table (Item 8). The current RAG retrieval:
1. Finds chunks mentioning "subscription" but not the specific footnote definition
2. The definitional boost helps but footnotes are embedded in table HTML, not standalone text

**Evidence**: The footnotes are structured like:
```html
(5) Includes annual and monthly fees associated with Amazon Prime memberships...
(6) Includes sales from healthcare services, licensing...
```

**Fix Required**:
1. Enhance `extract_footnote_ids_from_table()` to parse AMZN's footnote format (currently works for AAPL)
2. Pass footnote text directly to RAG generation as high-priority context
3. Add DOM-based footnote extraction (find `<sup>(N)</sup>` tags and trace to footnote definitions)

**Effort**: Medium (3-4 hours)  
**Impact**: Fixes AMZN (2 lines → 100% coverage)

---

### 1.3 GOOGL — "Google Network" Missing

**Line Missing**:
- `Google Network` — Manual: "Advertising served on Google Network properties participating in AdMob, AdSense, and Google Ad Manager"

**Root Cause**: The RAG query for "Google Network" returns chunks about Google's network infrastructure (data centers, bandwidth) rather than the advertising network definition. The definition is in MD&A Item 7 under "Google Advertising" as a bullet point.

**Evidence**: Manual inspection shows the definition is:
> "Google Network revenues consist of revenues generated primarily from ads placed on Google Network properties participating in AdMob, AdSense, and Google Ad Manager"

This is a short, bullet-point definition that may not score highly in semantic search.

**Fix Required**:
1. Add "AdMob AdSense Google Ad Manager" to query expansion for "Google Network"
2. Or: Add extractive patterns for bullet-point definitions in MD&A sections

**Effort**: Low (1 hour)  
**Impact**: Fixes GOOGL (1 line → 100% coverage)

---

### 1.4 AAPL Services — Incomplete Description

**Issue**: Current says "Services also include revenue from advertising, the App Store, and cloud services" which is incomplete.

**Manual has**: Full breakdown listing Advertising, AppleCare, Cloud Services, Digital Content (App Store, Books, Music, Video, Games, Podcasts), Subscription Services (Apple Arcade, Fitness+, Music, News+, TV), and Payment Services (Apple Card, Apple Pay).

**Root Cause**: The RAG is finding a generic reference to Services, not the structured Item 1 subsection that has 6 detailed paragraphs for each service category.

**Evidence**: Apple's 10-K has this structure in Item 1:
```
Services
The Company offers the following services:

Advertising
Advertising includes...

AppleCare
AppleCare includes...

Cloud Services
Cloud Services store and keep...

Digital Content
Digital Content includes the App Store...

Subscription Services
Subscription Services include...

Payment Services
Payment Services include...
```

**Fix Required**:
1. Enhance heading-based extraction to capture **all sub-headings** under a parent heading
2. Concatenate sub-section content when found
3. Increase context window for heading extraction

**Effort**: Medium (2-3 hours)  
**Impact**: Significantly improves AAPL Services description

---

## 2. Quality Assessment by Component

### 2.1 Description Quality Metrics

| Component | Current State | Target State |
|-----------|--------------|--------------|
| Specific product names | ~70% | 95% |
| Footnote extraction | ~40% | 90% |
| Sub-heading capture | ~20% | 80% |
| Accounting filter | 90% | 95% |
| Company language preservation | 85% | 95% |

### 2.2 Comparison: Current vs. Manual Inspection

**AAPL Services**:
```
CURRENT: "Services also include revenue from advertising, the App Store, and cloud services."

MANUAL: "Services include: Advertising (third-party licensing arrangements and Apple's own 
ad platforms); AppleCare (fee-based support, priority access to technical support, 
repair/replacement services, additional coverage in many cases); Cloud services (store and 
keep customer content up-to-date and available across Apple devices and Windows PCs); 
Digital content platforms incl. the App Store (discover/download apps and digital content 
such as books, music, video, games, podcasts); Subscription services (Apple Arcade, 
Fitness+, Music, News+, Apple TV offering original content and live sports); and Payment 
services (Apple Card and Apple Pay)."
```
**Gap**: ~80% of detail missing

**MSFT Server products and cloud services**:
```
CURRENT: "Server products and cloud services consists of Microsoft's public, private, 
and hybrid server products and cloud services, including Azure and other cloud services 
(such as cloud and AI consumption-based services, GitHub cloud services, Nuance Healthcare 
cloud services, and virtual desktop offerings), as well as on-premises server products 
like SQL Server, Windows Server, Visual Studio, System Center, and related Client Access 
Licenses."

MANUAL: "Cloud + server portfolio including Azure and other cloud services (cloud/AI 
consumption services, GitHub cloud services, Nuance Healthcare cloud services, virtual 
desktop offerings, other cloud services) and Server products (SQL Server, Windows Server, 
Visual Studio, System Center, related CALs, other on-prem offerings)."
```
**Gap**: Very close — acceptable quality ✅

---

## 3. Proposed Fix Tiers

### Tier 0: NVDA Table Extraction (Blocker)
**Must fix before any production run**

| Task | Description | File | Effort |
|------|-------------|------|--------|
| T0.1 | Add label_col validation (reject if >50% numeric) | `react_agents.py` | 1h |
| T0.2 | Add fallback: try column 0 if validation fails | `react_agents.py` | 30m |
| T0.3 | Test on NVDA, verify Compute/Networking/Gaming labels | `pipeline.py` | 30m |

### Tier 1: AMZN Footnote Extraction
**Needed for 100% AMZN coverage**

| Task | Description | File | Effort |
|------|-------------|------|--------|
| T1.1 | Parse AMZN's footnote format `(N) Includes...` | `react_agents.py` | 2h |
| T1.2 | Add footnote text as RAG high-priority context | `rag/generation.py` | 1h |
| T1.3 | Test Subscription services and Other get descriptions | `pipeline.py` | 30m |

### Tier 2: GOOGL Query Expansion
**Needed for 100% GOOGL coverage**

| Task | Description | File | Effort |
|------|-------------|------|--------|
| T2.1 | Add domain-specific terms to query builder | `rag/generation.py` | 1h |
| T2.2 | Add "AdMob AdSense Google Ad Manager" for "Google Network" | `rag/generation.py` | 30m |

### Tier 3: AAPL Services Enhancement
**Nice-to-have for richer descriptions**

| Task | Description | File | Effort |
|------|-------------|------|--------|
| T3.1 | Enhance heading extraction to capture sub-headings | `react_agents.py` | 2h |
| T3.2 | Concatenate sub-section content for parent labels | `react_agents.py` | 1h |
| T3.3 | Increase heading context window to 10000 chars | `react_agents.py` | 30m |

---

## 4. Architecture Observations

### 4.1 What's Working Well

1. **RAG definitional boost** — P1's `_boost_definitional_chunks()` fixed META's "Other revenue" problem by re-ranking chunks that contain both the label AND definition patterns.

2. **Note 2 classification** — Separating `note_revenue_sources` from `note_revenue_recognition` prevents accounting language contamination.

3. **Two-tier retrieval** — Table-local → full-filing fallback provides good precision/recall balance.

4. **Accounting filter** — `strip_accounting_sentences()` removes most revenue recognition language.

### 4.2 What Needs Improvement

1. **Footnote extraction** — Current implementation assumes footnotes are in visible text blocks. AMZN's footnotes are embedded in table HTML and need DOM-based extraction.

2. **Multi-section aggregation** — AAPL Services has 6 sub-sections that should be concatenated. Current heading extraction stops at first match.

3. **Label column validation** — No check that the inferred label_col actually contains text labels, not numbers.

4. **Query specialization** — Generic queries like "Google Network products and services" miss domain-specific terminology (AdMob, AdSense).

---

## 5. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NVDA fix breaks other tickers | Low | High | Test all 6 tickers after each change |
| Footnote extraction too aggressive | Medium | Medium | Add footnote confidence scoring |
| Sub-heading aggregation includes noise | Medium | Low | Limit to 3 sub-sections max |
| RAG query expansion reduces precision | Low | Medium | A/B test with current queries |

---

## 6. Recommended Execution Order

```
Phase 1: Blockers (2-3 hours)
├── T0.1-T0.3: Fix NVDA table extraction
└── Verify all 6 tickers still work

Phase 2: Coverage (4-5 hours)
├── T1.1-T1.3: Fix AMZN footnotes
├── T2.1-T2.2: Fix GOOGL Google Network
└── Verify coverage: AMZN 100%, GOOGL 100%

Phase 3: Quality (3-4 hours)
├── T3.1-T3.3: Enhance AAPL Services
└── Verify description richness

Phase 4: Validation
├── Full run on all 6 tickers
├── Compare against manual inspection files
└── Generate final CSV1 for review
```

---

## 7. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Line coverage | 82.1% (32/39) | 100% |
| Description coverage | 69.2% (27/39) | 95%+ |
| NVDA functional | No | Yes |
| AMZN 100% | No | Yes |
| GOOGL 100% | No | Yes |

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `revseg/react_agents.py` | T0.1-T0.2, T1.1, T3.1-T3.3 |
| `revseg/rag/generation.py` | T1.2, T2.1-T2.2 |
| `revseg/pipeline.py` | Minimal (test harness) |

---

## 9. Decision Required

**Proceed with Tier 0 (NVDA fix) immediately?**

This is a blocker — without it, NVDA shows 0 valid lines. The fix is low-risk and quick.

After Tier 0, I recommend:
1. Re-run all 6 tickers
2. Validate NVDA now shows Compute/Networking/Gaming/ProViz/Automotive/OEM
3. Then proceed to Tier 1 (AMZN) and Tier 2 (GOOGL)

---

## Appendix A: Manual vs. Current Comparison Table

| Ticker | Line | Manual Description | Current Description | Status |
|--------|------|-------------------|---------------------|--------|
| AAPL | iPhone | Apple's line of smartphones based on iOS... | iPhone revenue consists of sales of Apple's iPhone line... | ✅ OK |
| AAPL | Mac | MacBook Air, MacBook Pro, iMac, Mac mini... | Line of personal computers based on macOS... | ⚠️ Missing models |
| AAPL | iPad | iPad Pro, iPad Air, iPad, iPad mini | iPad devices including iPad Air, iPad mini... | ✅ OK |
| AAPL | Wearables | Apple Watch, AirPods, Apple Vision Pro, Apple TV, HomePod | AirPods, AirPods Pro, AirPods Max, Beats, Apple Vision Pro... | ✅ OK |
| AAPL | Services | 6 detailed sub-categories | "Also include advertising, App Store, cloud services" | ❌ Incomplete |
| AMZN | Online stores | Wide selection of consumable and durable goods | Sales through Amazon's online stores | ✅ OK |
| AMZN | Physical stores | Customers physically select items | Sales in physical retail locations | ✅ OK |
| AMZN | Third-party | Commissions, fulfillment, shipping fees | Programs that enable sellers to grow... | ✅ OK |
| AMZN | Advertising | Sponsored ads, display, video advertising | Advertising solutions to sellers, vendors... | ✅ OK |
| AMZN | Subscription | Amazon Prime, digital video, audiobooks... | (empty) | ❌ Missing |
| AMZN | AWS | Compute, storage, database, analytics, ML | Broad set of on-demand technology services... | ✅ OK |
| AMZN | Other | Healthcare, video licensing, shipping... | (empty) | ❌ Missing |
| GOOGL | Google Search | Advertising on Google Search properties | Revenues generated on Google search properties | ✅ OK |
| GOOGL | YouTube ads | Advertising on YouTube properties | Advertising revenues from ads on YouTube | ✅ OK |
| GOOGL | Google Network | AdMob, AdSense, Google Ad Manager | (empty) | ❌ Missing |
| GOOGL | Subscriptions | YouTube TV, Music, Google One, Pixel | Consumer subscriptions, platform revenues, device revenues | ✅ OK |
| GOOGL | Google Cloud | GCP, AI, Workspace | Infrastructure, platform services, applications | ✅ OK |
| GOOGL | Other Bets | Healthcare, internet services | Healthcare-related and internet services | ✅ OK |
| META | Advertising | Ads on Facebook, Instagram, Messenger | Revenue from selling advertising placements | ✅ OK |
| META | Other | WhatsApp Business, Meta Verified, Payments | WhatsApp Business Platform, Meta Verified... | ✅ OK |
| META | Reality Labs | Meta Quest, Ray-Ban Meta AI glasses | Consumer hardware products, software, content | ✅ OK |
| MSFT | (all 10 lines) | Detailed segment descriptions | Very close to manual | ✅ OK |
| NVDA | Compute | Data center compute for AI | "$116,193" (wrong label) | ❌ Bug |
| NVDA | Networking | InfiniBand, Ethernet platforms | "$82,875" (wrong label) | ❌ Bug |
| NVDA | Gaming | GeForce GPUs, GeForce NOW | (not extracted) | ❌ Bug |
| NVDA | ProViz | Quadro/RTX GPUs, Omniverse | (not extracted) | ❌ Bug |
| NVDA | Automotive | DRIVE platform for AV/EV | (not extracted) | ❌ Bug |
| NVDA | OEM and Other | No explicit definition in filing | (not extracted) | ❌ Bug |

---

## Appendix B: Current RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG Description Extraction                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CHUNKING (chunking.py)                                      │
│     ├── Structure-aware: by headings/notes                      │
│     ├── Section metadata: item1, note_revenue, etc.             │
│     ├── P1: note_revenue → note_revenue_sources OR              │
│     │                      note_revenue_recognition             │
│     └── TOC detection (non-destructive tagging)                 │
│                                                                  │
│  2. INDEXING (index.py)                                         │
│     ├── OpenAI text-embedding-3-small                           │
│     ├── FAISS IndexFlatIP (inner product)                       │
│     ├── Two-tier: table-local → full-filing                     │
│     └── BLOCKED_SECTIONS: {note_revenue_recognition}            │
│                                                                  │
│  3. RETRIEVAL (index.py)                                        │
│     ├── Embed query with build_rag_query()                      │
│     ├── Search tier1 (local) first                              │
│     ├── Fallback to tier2 (full) if threshold not met           │
│     ├── P1: _boost_definitional_chunks() re-ranks               │
│     └── Return top-k chunks with scores                         │
│                                                                  │
│  4. GENERATION (generation.py)                                  │
│     ├── Evidence gate: require preferred section OR high score  │
│     ├── Extract candidate products (extractive-first)           │
│     ├── LLM call with strict JSON schema                        │
│     └── Validate quotes exist in chunks                         │
│                                                                  │
│  5. QA (qa.py)                                                  │
│     └── Write csv1_desc_coverage.json artifact                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

*End of DEV_PROPOSAL.md*
