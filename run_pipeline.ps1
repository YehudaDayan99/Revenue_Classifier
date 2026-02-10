# Revenue Classifier Pipeline - Run Script
# Sets required environment variables and runs the pipeline

# Required environment variables
$env:SEC_USER_AGENT = "RevenueClassifier/1.0 (yehud.dayan@outlook.com)"
$env:OPENAI_API_KEY_FILE = "C:\Users\yehud\OneDrive\DataServices\DoubleRobust\API Keys\Business_Segment_V1.txt"

# Default parameters (can be overridden via command line)
$tickers = if ($args[0]) { $args[0] } else { "NVDA,AAPL,MSFT,GOOGL,AMZN,META" }
$outDir = if ($args[1]) { $args[1] } else { "data/outputs" }

Write-Host "Running Revenue Classifier Pipeline"
Write-Host "  Tickers: $tickers"
Write-Host "  Output:  $outDir"
Write-Host ""

python -m revseg.pipeline --tickers $tickers --csv1-only --out-dir $outDir
