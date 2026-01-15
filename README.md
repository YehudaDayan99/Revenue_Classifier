# Business Classification - Revenue Segmentation

A Python pipeline for extracting and analyzing revenue segmentation data from SEC 10-K filings.

## Overview

This project implements a multi-stage pipeline to:
1. **Download 10-K filings** from SEC EDGAR for US-traded public companies
2. **Extract table candidates** from HTML filings
3. **Identify revenue segmentation tables** (future: LLM-based selection)
4. **Extract structured revenue data** (future: CSV generation)

## Project Structure

```
.
├── revseg/                    # Core modules
│   ├── sec_edgar.py           # SEC EDGAR download functionality
│   ├── table_candidates.py    # Table extraction from HTML
│   └── *.md                   # Documentation
├── Base.ipynb                 # Main notebook for running pipeline
├── data/                      # Downloaded filings and outputs (gitignored)
└── .cache/                    # Cache directory (gitignored)
```

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Mac/Linux
```

2. Install dependencies:
```bash
pip install requests beautifulsoup4 lxml
```

3. Set SEC compliance environment variable:
```bash
# Windows PowerShell
$env:SEC_USER_AGENT="YourApp/0.1 (your.email@domain.com)"

# Windows CMD
set SEC_USER_AGENT=YourApp/0.1 (your.email@domain.com)

# Mac/Linux
export SEC_USER_AGENT="YourApp/0.1 (your.email@domain.com)"
```

Or set it in your notebook:
```python
import os
os.environ["SEC_USER_AGENT"] = "YourApp/0.1 (your.email@domain.com)"
```

## OpenAI API key (for ReAct pipeline)

The ReAct pipeline uses the OpenAI API and expects `OPENAI_API_KEY` in your environment.

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Important: do not commit API keys into notebooks or source files.

## Usage

See `Base.ipynb` for example usage. The notebook demonstrates:
- Downloading 10-K filings for multiple tickers
- Extracting table candidates from HTML
- Ranking and inspecting candidate tables

## Documentation

- `revseg/10k Extraction Readme.md` - Stage 1: SEC EDGAR download
- `revseg/Table Candidate readme.md` - Stage 3: Table extraction

## License

[Add your license here]
