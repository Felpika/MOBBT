
import requests
import streamlit as st
import yfinance as yf
import pandas as pd
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

@st.cache_data(ttl=600, show_spinner=False)
def get_asset_price_current(ticker):
    """Busca preço ATUAL do ativo via yfinance"""
    try:
        full_ticker = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        stock = yf.Ticker(full_ticker)
        data = stock.history(period="1d")
        
        if data.empty:
            return 0.0
            
        # Garante que não é MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return 0.0
    except:
        return 0.0

@st.cache_data(ttl=600, show_spinner=False)
def get_asset_price_yesterday(ticker):
    """Busca preço de FECHAMENTO DE ONTEM do ativo (mesmo dia da B3 API)"""
    try:
        full_ticker = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        stock = yf.Ticker(full_ticker)
        # Busca últimos 5 dias para garantir ter dados
        data = stock.history(period="5d")
        
        if data.empty:
            return 0.0
            
        # Garante que não é MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Remove linhas com NaN no Close
        data = data.dropna(subset=['Close'])
        
        if len(data) >= 2:
            # Retorna penúltimo fechamento (ontem)
            return float(data['Close'].iloc[-2])
        elif len(data) == 1:
            # Se só tem 1 dia (ex: feriado recente), retorna o único disponível
            return float(data['Close'].iloc[-1])
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
    # Updated logic:
    # If strike < 100, multiply by 10 (e.g. 41 -> 410) to reach 3 digits.
    # If strike >= 100, use as is (e.g. 161 -> 161), presumably already 3 digits.
    if strike < 100:
        strike_val = int(strike * 10)
    else:
        strike_val = int(strike)
        
    return f"{asset_code}{month_letter}{strike_val}"
