# Probabilistic Crypto Forecasting Engine

A specialized forecasting system that generates probabilistic price paths for crypto assets (BTC, ETH, SOL) and Gold (XAU) using GARCH(1,1) volatility modeling and Student's t-distribution simulations.

## Features

- **Advanced Modeling**: Uses GARCH(1,1) to capture volatility clustering and `Student's t` distribution for fat-tailed risk modeling.
- **Monte Carlo Simulation**: Generates 1000 independent price paths per asset for a 24-hour horizon (288 steps @ 5min intervals).
- **Robust Data**: Fetches real-time data from Binance. Implements a smart fallback for Gold (XAU) that attempts Alpha Vantage first but seamlessly switches to Paxos Gold (PAXG) on Binance if API limits are hit.
- **Strict Output**: Produces a strictly formatted JSON output suitable for ingestion by frontend visualization tools.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main orchestrator:

```bash
python main.py
```

This will generate a `predictions.json` file in the root directory.

### Configuration

- **Alpha Vantage API**: Optional. Set `ALPHA_VANTAGE_KEY` in your environment variables to use real forex data for XAU. If not set, the system uses the high-correlation PAXG/USDT proxy.

## Project Structure

- `main.py`: Entry point. Orchestrates data fetching, modeling, and formatting.
- `model_engine.py`: Contains the econometrics logic (GARCH fitting and simulation).
- `data_fetcher.py`: Handles API connectivity and error recovery.
- `verify_requirements.py`: Utility script to validate output format and constraints.
