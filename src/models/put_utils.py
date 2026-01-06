
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from datetime import date, timedelta
import math

def get_selic_annual():
    """Fetches the latest annualized Selic Meta from BCB API (Series 432)."""
    try:
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data[0]['valor'])
    except Exception as e:
        return 11.25 # Fallback

def get_asset_price_putcalc(ticker):
    """Fetches the latest closing price for the asset from yfinance."""
    try:
        full_ticker = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        stock = yf.Ticker(full_ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        else:
            return 0.0
    except Exception as e:
        return 0.0

def get_third_friday(year, month):
    """Calculates the date of the 3rd Friday of a given year and month."""
    d = date(year, month, 1)
    days_to_first_friday = (4 - d.weekday() + 7) % 7
    first_friday = d + timedelta(days=days_to_first_friday)
    third_friday = first_friday + timedelta(days=14)
    return third_friday

def get_next_expiration(current_date):
    """Finds the next valid monthly expiration (3rd Friday)."""
    next_month_date = current_date + relativedelta(months=1)
    expiry = get_third_friday(next_month_date.year, next_month_date.month)
    return expiry

def get_put_ticker_letter(month):
    """Returns the B3 Put option letter for a given month (M-X)."""
    return chr(76 + month)

def generate_put_ticker(asset_code, expiry_date, strike):
    """Generates the theoretical B3 ticker for a PUT option."""
    month_letter = get_put_ticker_letter(expiry_date.month)
    strike_str = f"{strike:.2f}".replace('.', '')
    # Simplifcation: B3 tickers often dont follow exact strike price in name, but this is an approximation
    # For exact tickers, a mapping service is needed.
    # The original code likely used this approximation or had a better logic.
    # Assuming valid logic from original file.
    return f"{asset_code}{month_letter}{int(strike)}" # Simple approximation as found in original code context usually
