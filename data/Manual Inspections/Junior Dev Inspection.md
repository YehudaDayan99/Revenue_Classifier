I scanned the three attached 10-K HTMLs and extracted **only revenue products/services with an explicit revenue figure for the latest year** (plus any explicitly reported revenue adjustment line needed for the total to reconcile to 100%).

Units: **USD millions** (as presented in the filings).

Final Output 

## Revenue line items (latest year, sums to 100% per company)

| Year | Company   | Ticker | Segment                             | Item                                                 | Income $ (USD m) | Income % |
| ---: | --------- | ------ | ----------------------------------- | ---------------------------------------------------- | ---------------: | -------: |
| 2025 | Microsoft | MSFT   | Intelligent Cloud                   | Server products and cloud services                   |           98,435 |  34.9402 |
| 2025 | Microsoft | MSFT   | Intelligent Cloud                   | Enterprise and partner services                      |            7,760 |   2.7545 |
| 2025 | Microsoft | MSFT   | Productivity and Business Processes | Microsoft 365 Commercial products and cloud services |           87,767 |  31.1535 |
| 2025 | Microsoft | MSFT   | Productivity and Business Processes | LinkedIn                                             |           17,812 |   6.3225 |
| 2025 | Microsoft | MSFT   | Productivity and Business Processes | Dynamics products and cloud services                 |            7,827 |   2.7783 |
| 2025 | Microsoft | MSFT   | Productivity and Business Processes | Microsoft 365 Consumer products and cloud services   |            7,404 |   2.6281 |
| 2025 | Microsoft | MSFT   | More Personal Computing             | Gaming                                               |           23,455 |   8.3255 |
| 2025 | Microsoft | MSFT   | More Personal Computing             | Windows and Devices                                  |           17,314 |   6.1457 |
| 2025 | Microsoft | MSFT   | More Personal Computing             | Search and news advertising                          |           13,878 |   4.9261 |
| 2025 | Microsoft | MSFT   | Other                               | Other                                                |               72 |   0.0256 |
| 2024 | Alphabet  | GOOGL  | Google Services                     | Google Search & other                                |          198,084 |  56.5925 |
| 2024 | Alphabet  | GOOGL  | Google Services                     | YouTube ads                                          |           36,147 |  10.3272 |
| 2024 | Alphabet  | GOOGL  | Google Services                     | Google Network                                       |           30,359 |   8.6736 |
| 2024 | Alphabet  | GOOGL  | Google Services                     | Google subscriptions, platforms, and devices         |           40,340 |  11.5251 |
| 2024 | Alphabet  | GOOGL  | Google Cloud                        | Google Cloud                                         |           43,229 |  12.3505 |
| 2024 | Alphabet  | GOOGL  | Other Bets                          | Other Bets                                           |            1,648 |   0.4708 |
| 2024 | Alphabet  | GOOGL  | Other                               | Hedging gains (losses)                               |              211 |   0.0603 |
| 2025 | Apple     | AAPL   | Apple                               | iPhone                                               |          209,586 |  50.3618 |
| 2025 | Apple     | AAPL   | Apple                               | Mac                                                  |           33,708 |   8.0997 |
| 2025 | Apple     | AAPL   | Apple                               | iPad                                                 |           28,023 |   6.7337 |
| 2025 | Apple     | AAPL   | Apple                               | Wearables, Home and Accessories                      |           35,686 |   8.5750 |
| 2025 | Apple     | AAPL   | Apple                               | Services                                             |          109,158 |  26.2298 |

Notes on interpretation:

* **MSFT**: the filing’s disaggregation table lists the product/service lines directly (plus “Other”); it does **not** label segments in that table, so the Segment assignment above uses a deterministic mapping consistent with MSFT’s segment definitions elsewhere in the 10-K (this is straightforward to codify).
* **GOOGL**: to ensure **100% reconciliation**, the table includes **“Hedging gains (losses)”** as part of total revenues. It is not a product/service, but it is an explicit revenue line and is required to reconcile to the filing’s “Total revenues.”
* **AAPL**: Apple is effectively one reportable segment; the explicit revenue lines are the **net sales by product category**.

---

## How to execute this reliably in your code base (agent/ReAct-friendly)

To make this robust inside your pipeline, treat “revenue line item extraction” as a **deterministic tool** the agent calls, with LLM used only for edge-case disambiguation:

1. **Table discovery (deterministic)**

   * Parse all HTML tables (`pandas.read_html` or custom BeautifulSoup table parser).
   * Score candidate tables by:

     * presence of the latest year (e.g., 2025/2024) in header rows,
     * presence of “Total revenue / Total net sales / Total revenues” row,
     * overlap with known keywords (“Google Search”, “iPhone”, “Server products and cloud services”, etc.).
   * Select the best table by score, then keep its **table index** as evidence.

2. **Row selection and subtotal suppression (deterministic rules)**

   * Keep rows with explicit numeric revenue in the latest year column.
   * Drop subtotals (rules like: contains “total”, “advertising”, “services total”, etc.) unless the company has no finer disaggregation.
   * Keep small explicit residual lines (e.g., MSFT “Other”) to reconcile to 100.

3. **Segment assignment**

   * If segment is present in-table, use it directly.
   * If not (MSFT case), apply a maintained mapping dictionary:

     * `item -> segment` (stable over time, easy to version).
   * Only use LLM if a previously unseen item appears and cannot be mapped with heuristics.

4. **Reconciliation gate**

   * Compute % of total and assert `abs(sum(pct) - 100) < tolerance`.
   * If it fails, fall back to:

     * include one level higher aggregation (segment totals) OR
     * include explicit adjustment lines (as Alphabet does with hedging).

5. **Evidence object (critical for your Tab 3)**

   * Store: filing id, table index, extracted row text, raw numeric cell strings, and normalized numeric value.
   * This is what your “evidence” tab should display.
