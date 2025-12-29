import requests
import pandas as pd
import numpy as np
import os
import time

class DataFetcher:
    def __init__(self):
        self.binance_url = "https://api.binance.com/api/v3/klines"
        # XAU (Gold) fallback if Alpha Vantage fails/no key
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        # PASTE YOUR ALPHA VANTAGE KEY HERE
        self.alpha_vantage_key = "HL2PBYPWN1RMK2WU" 
        
        # Fallback to environment variable if the above is still the placeholder
        if self.alpha_vantage_key == "PASTE_YOUR_KEY_HERE":
             self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "DEMO_KEY")

        # Map project symbols to API symbols
        self.symbol_map = {
            "BTC": "BTCUSDT",
            "ETH": "ETHUSDT",
            "SOL": "SOLUSDT",
            "XAU": "XAUUSD" # For Alpha Vantage, might be different or require physical currency API
        }

    def fetch_crypto_data(self, symbol, interval="5m", limit=1000):
        """
        Fetch historical data from Binance.
        Returns a DataFrame with 'close' price and 'timestamp'.
        """
        api_symbol = self.symbol_map.get(symbol, f"{symbol}USDT")
        params = {
            "symbol": api_symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(self.binance_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Binance Klines: [Open time, Open, High, Low, Close, Volume, ...]
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "number_of_trades", 
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            
            # Convert types
            df["timestamp"] = pd.to_numeric(df["timestamp"])
            df["close"] = pd.to_numeric(df["close"])
            
            # Just return needed columns
            return df[["timestamp", "close"]]
        
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_gold_data(self):
        """
        Fetch Gold (XAU/USD) data. 
        Tries Alpha Vantage, falls back to a mock history if allowed/needed, 
        but strictly requesting 'real' data logic.
        """
        # Note: Free Alpha Vantage keys are rate limited (5req/min, 500/day).
        # Function: TIME_SERIES_INTRADAY
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": "XAUUSD", # Often works for forex/crypto on AV? actually for FOREX it's better to use FX_INTRADAY maybe? 
            # XAU is treated as currency in many APIs. Let's try FX_INTRADAY or just mimic crypto if it fails.
            # AlphaVantage doesn't always have XAUUSD widely available on free tier cleanly.
            # Alternative: Binance actually lists PAXGUSDT (Pax Gold) which tracks gold price. 
            # This is often a safer bet for a reliable 'real' gold price feed in a crypto infrastructure.
            # Let's try fetching PAXGUSDT from Binance as a proxy for XAU first for reliability.
            "interval": "5min",
            "apikey": self.alpha_vantage_key
        }
        
        try:
            print(f"Attempting to fetch Gold data from Alpha Vantage...")
            response = requests.get(self.alpha_vantage_url, params=params)
            # Alpha Vantage returns 200 even for errors sometimes, check json content
            data = response.json()
            
            if "Error Message" in data or "Note" in data or "Information" in data:
                # 'Note' usually means rate limit
                raise Exception(f"Alpha Vantage API Error/Limit: {data}")
            
            # Parse Time Series
            # Format: 'Time Series (5min)': { '2023-10...': { '1. open': ... } }
            key_name = list(data.keys())[1] # Usually "Time Series (5min)" or "Time Series FX (5min)"
            ts_data = data[key_name]
            
            records = []
            for timestamp, values in ts_data.items():
                records.append({
                    "timestamp": timestamp, # Need to convert to unix? Or pandas handles it
                    "close": float(values["4. close"])
                })
                
            df = pd.DataFrame(records)
            # Sort chronological
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            
            # Need strict format for model engine?
            # Model engine expects numerical timestamps if used for plotting, but mainly just 'close' column
            return df[["timestamp", "close"]]

        except Exception as e:
            print(f"⚠️ Alpha Vantage failed ({e}). Falling back to PAXGUSDT (Binance Proxy) for reliability...")
            return self.fetch_crypto_data("PAXG", interval="5m", limit=1000)

    def get_latest_price(self, symbol):
        """
        Get the specific latest price.
        """
        df = None
        if symbol == "XAU":
            df = self.fetch_gold_data()
        else:
            df = self.fetch_crypto_data(symbol)
            
        if df is not None and not df.empty:
            return df.iloc[-1]["close"]
        return 0.0

    def get_history(self, symbol, limit=1000):
        if symbol == "XAU":
            return self.fetch_gold_data()
        return self.fetch_crypto_data(symbol, limit=limit)
