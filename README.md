# Revenue Classifier

A Python pipeline for extracting revenue line items and their descriptions from SEC 10-K filings using LLM-assisted agents.

## Overview

This project extracts structured revenue data from public company 10-K filings:
- **Revenue line items** with dollar amounts (e.g., iPhone: $209,586M)
- **Revenue groups** mapping items to reportable segments
- **Company-language descriptions** explaining what each line item represents

## Quick Start

```powershell
# Install dependencies
pip install requests beautifulsoup4 lxml openai

# Option 1: Use the startup script (recommended)
.\run_pipeline.ps1

# Option 2: Manual setup
$env:SEC_USER_AGENT = "YourApp/0.1 (your.email@domain.com)"
$env:OPENAI_API_KEY_FILE = "path/to/your/api_key.txt"  # File containing API key
python -m revseg.pipeline --tickers MSFT,AAPL,GOOGL --csv1-only
```

## Output

The pipeline generates `csv1_segment_revenue.csv`:

| Company Name | Ticker | Fiscal Year | Revenue Group | Revenue Line | Description | Revenue ($m) |
|--------------|--------|-------------|---------------|--------------|-------------|--------------|
| Apple Inc. | AAPL | 2025 | Product/Service | iPhone | iPhone is the Company's line of smartphones... | 209,586.0 |
| Apple Inc. | AAPL | 2025 | Product/Service | Services | Advertising, AppleCare, Cloud Services... | 109,158.0 |

## Project Structure

```
Revenue Classifier/
├── revseg/                    # Core modules
│   ├── pipeline.py            # Main orchestration
│   ├── react_agents.py        # LLM agents for table/description extraction
│   ├── table_candidates.py    # HTML table extraction
│   ├── mappings.py            # Company-specific segment mappings
│   ├── extraction/            # Deterministic extraction logic
│   └── rag/                   # Optional RAG-based descriptions
├── docs/
│   └── PIPELINE_FLOW.md       # Detailed technical documentation
├── data/
│   ├── 10k/                   # Downloaded SEC filings
│   └── outputs/               # Generated CSV files
└── tests/                     # Unit tests
```

## Command Line Options

```bash
python -m revseg.pipeline --tickers MSFT,AAPL --csv1-only [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | (required) | Comma-separated tickers |
| `--out-dir` | `data/outputs` | Output directory |
| `--csv1-only` | `false` | Generate only CSV1 (recommended) |
| `--use-rag` | `false` | Use RAG for descriptions |
| `--model-fast` | `gpt-4.1-mini` | Model for table selection |
| `--model-quality` | `gpt-4.1` | Model for descriptions |

## Current Coverage (6 Tickers Validated)

| Ticker | Lines | Validation | Notes |
|--------|-------|------------|-------|
| NVDA | 6 | ✅ 0.00% | All segments extracted |
| AAPL | 5 | ✅ 0.00% | Product categories |
| MSFT | 10 | ✅ 0.00% | Mapped to 3 segments |
| GOOGL | 7 | ✅ 0.00% | Includes hedging adjustment |
| AMZN | 7 | ✅ 0.00% | Product/service disaggregation |
| META | 2 | ✅ 0.00% | Advertising + Other revenue |

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | Non-technical + technical overview |
| [`docs/PIPELINE_FLOW.md`](docs/PIPELINE_FLOW.md) | Detailed technical reference |

The Pipeline Flow includes:
- Architecture diagram
- LLM agent descriptions
- Regex patterns used
- Data structures
- Troubleshooting guide

## License

[Add your license here]
