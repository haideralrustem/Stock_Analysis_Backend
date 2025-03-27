import os
import pandas as pd
import datetime
import time
import numpy as np

import yfinance as yf




def get_sp_500_stocks():

  # Step 1: Get S&P 500 tickers from Wikipedia
  url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
  sp500_table = pd.read_html(url)[0]  # Read first table from Wikipedia page
  sp500_tickers = sp500_table["Symbol"].tolist()  # Get tickers as a list

  # Dictionary to store historical data
  sp500_data = {}

  # Step 2: Loop through each stock and get 5 years of data
  for ticker in sp500_tickers:
    try:
      print(f"Fetching data for {ticker}...")
      stock = yf.Ticker(ticker)
      df = stock.history(period="5y")  # Get 5 years of historical data
      df = df.reset_index()
      df['Adj Close'] = df['Close']
      df = df [['Date','Open','High','Low','Close','Adj Close','Volume']]
      df['Date'] = pd.to_datetime(df['Date'])
      # Convert to date only (remove the time part)
      df['Date'] = df['Date'].dt.date
      sp500_data[ticker] = df
      time.sleep(1)  # Avoid hitting API rate limits
    except Exception as e:
      print(f"Error fetching {ticker}: {e}")

  return

#________________________________________



if __name__ == "__main__":
  print()

  get_sp_500_stocks()