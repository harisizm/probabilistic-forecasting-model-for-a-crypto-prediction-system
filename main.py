import time
import json
import numpy as np
from data_fetcher import DataFetcher
from model_engine import ForecastModel

def format_price(price):
    """
    Format price to MAX 8 digits total (including decimals).
    """
    if price == 0:
        return 0
    
    s = "{:.10f}".format(price)
    
    # Split into integer and decimal parts
    if '.' in s:
        parts = s.split('.')
        int_part = parts[0]
        dec_part = parts[1]
        
        # Calculate allowed decimal places
        # Total digits = len(int_part) + len(dec_part)
        # We want len(int_part) + len(dec_part) + (1 for dot) <= 8?
        # "8 digits total (including decimals)" could mean string length <= 8.
        # e.g. 1234.567 (8 chars)
        
        allowed_len = 8
        if len(int_part) >= allowed_len:
            return round(price) # No decimals fitting
        
        remaining = allowed_len - len(int_part) - 1 # -1 for dot
        
        if remaining > 0:
             # Truncate string to avoid rounding up complexity issues for max len
             # But 'round' is safer for values.
             # Let's try flexible rounding.
             return round(price, remaining)
        else:
             return round(price)
            
    return price

def run_prediction_system():
    assets = ["BTC", "ETH", "SOL", "XAU"]
    
    fetcher = DataFetcher()
    model = ForecastModel()
    
    results = {}
    
    # Common start timestamp for all paths (current time)
    start_timestamp = int(time.time())
    
    print(f"Starting Prediction System at {start_timestamp}")
    print("-" * 50)
    
    for asset in assets:
        print(f"Processing {asset}...")
        
        # 1. Fetch Data
        # We need enough history for GARCH to stabilize. 
        # 1000 5m candles = ~3.5 days. Should be enough.
        prices_df = fetcher.get_history(asset, limit=1000)
        
        if prices_df is None or len(prices_df) < 100:
            print(f"⚠️ Insufficient data for {asset}, skipping.")
            continue
            
        current_price = prices_df.iloc[-1]["close"]
        print(f"  Current Price: {current_price}")
        
        # 2. Run Model
        # Returns 1000 paths of 288 points
        paths = model.fit_and_simulate(prices_df, current_price=current_price)
        
        # 3. Format Output
        # [ start_timestamp, 300, [path1], [path2]... ]
        
        formatted_paths = []
        for path in paths:
            # Format each price in the path
            formatted_p = [format_price(p) for p in path]
            formatted_paths.append(formatted_p)
            
        asset_output = [
            start_timestamp,
            300, # 5 min in seconds
            *formatted_paths # Unpack the 1000 paths into the list
        ]
        
        results[asset] = asset_output
        print(f"  ✅ Generated {len(formatted_paths)} paths for {asset}")
        
    return results

if __name__ == "__main__":
    output = run_prediction_system()
    
    # Save to file
    with open("predictions.json", "w") as f:
        json.dump(output, f, indent=None) # No indent to keep file smaller? Or consistent with user needs.
        # User example showed indentation.
        
    print("-" * 50)
    print("Predictions saved to predictions.json")
