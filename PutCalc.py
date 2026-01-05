import streamlit as st
import pandas as pd
import math
import yfinance as yf
import requests
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

# --- Configuration ---
st.set_page_config(page_title="Calculadora Yield Put Writing", layout="wide")

# --- Helper Functions ---

def get_selic_annual():
    """Fetches the latest annualized Selic Meta from BCB API (Series 432)."""
    try:
        # API BCB Series 432 (Taxa de juros - Meta Selic definida pelo Copom)
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data[0]['valor'])
    except Exception as e:
        st.error(f"Erro ao buscar Selic: {e}. Usando valor padrão 11.25%")
        return 11.25 # Fallback

def get_asset_price(ticker):
    """Fetches the latest closing price for the asset from yfinance."""
    try:
        # Append .SA for Brazilian stocks if not present
        full_ticker = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
        stock = yf.Ticker(full_ticker)
        # Get fast info first (often has 'currentPrice' or 'regularMarketPrice')
        info = stock.info # This can be slow sometimes, let's try history
        # History is often more reliable
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

def generate_ticker(root, expiry_date, strike):
    """Generates a theoretical B3 ticker: ROOT + LETTER + STRIKE_INT."""
    letter = get_put_ticker_letter(expiry_date.month)
    strike_str = str(int(strike))
    return f"{root}{letter}{strike_str}"

# --- UI Setup ---

st.title("🛡️ Calculadora de Yield - Put Writing")

# Sidebar for Status
st.sidebar.header("Dados de Mercado (Automático)")

# Fetch Selic
with st.spinner("Buscando Taxa Selic..."):
    selic_annual = get_selic_annual()
selic_monthly = (pow(1 + selic_annual/100, 1/12) - 1) * 100

st.sidebar.metric("Selic Meta (a.a)", f"{selic_annual:.2f}%")
st.sidebar.metric("Selic (a.m estimada)", f"{selic_monthly:.4f}%")

# Main Inputs
col1, col2, col3 = st.columns(3)

with col1:
    asset_ticker = st.text_input("Ativo Objeto", value="BOVA11").upper()
    current_date = st.date_input("Data Base", value=date.today())

with col2:
    # Trigger auto fetch based on ticker
    if asset_ticker:
        with st.spinner(f"Buscando preço {asset_ticker}..."):
            fetched_price = get_asset_price(asset_ticker)
        
        if fetched_price > 0:
            asset_price = fetched_price
            st.success(f"Preço Atual: R$ {asset_price:.2f}")
        else:
            asset_price = st.number_input("Preço do Ativo (Manual - Falha na Busca)", value=0.0, step=0.01, format="%.2f")
    else:
         asset_price = st.number_input("Preço do Ativo (R$)", value=0.0, step=0.01, format="%.2f")

    collateral = st.number_input("Colateral Disponível (R$)", value=31018.00, step=100.0, format="%.2f")

# Logic
expiry = get_next_expiration(current_date)
suggested_strike = round(asset_price, 0)
suggested_ticker = generate_ticker(asset_ticker[:4], expiry, suggested_strike)

with col3:
    st.markdown("### Sugestão Automática")
    if asset_price > 0:
        st.info(f"Vencimento: **{expiry.strftime('%d/%m/%Y')}**")
        st.info(f"Strike ATM: **R$ {suggested_strike:.2f}**")
        st.info(f"Ticker: **{suggested_ticker}**")
    else:
        st.warning("Aguardando preço do ativo...")

st.markdown("---")

# Option Specifics Input
st.subheader("Dados da Opção")
c_op1, c_op2, c_op3 = st.columns(3)

with c_op1:
    selected_strike = st.number_input("Strike Selecionado", value=suggested_strike, step=0.5, format="%.2f")

with c_op2:
    actual_ticker = generate_ticker(asset_ticker[:4], expiry, selected_strike)
    st.text_input("Código da Opção (Teórico)", value=actual_ticker, disabled=True)
    
with c_op3:
    option_price = st.number_input("Preço da Put (Prêmio)", value=2.62, step=0.01, format="%.2f")

# --- Calculations ---

if selected_strike > 0 and option_price > 0 and asset_price > 0:
    qty_contracts = math.floor(collateral / selected_strike)
    notional = qty_contracts * selected_strike
    yield_pct = (option_price / asset_price) * 100 
    pct_cdi = (yield_pct / selic_monthly) * 100
    total_credit = qty_contracts * option_price

    # --- Display Results ---
    st.markdown("## Resultados")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Yield (Retorno)", f"{yield_pct:.2f}%")
    m2.metric("% do CDI", f"{pct_cdi:.0f}%")
    m3.metric("Qtd Contratos", f"{qty_contracts}")
    m4.metric("Crédito Total", f"R$ {total_credit:,.2f}")

    results_data = {
        "Métrica": ["Preço Ativo", "Selic a.a", "Preço Put", "Yield", "% CDI", "Qtd Contratos", "Notional/Colateral", "Valor Venda Put"],
        "Valor": [
            f"R$ {asset_price:.2f}",
            f"{selic_annual:.2f}%",
            f"R$ {option_price:.2f}",
            f"{yield_pct:.2f}%",
            f"{pct_cdi:.0f}%",
            f"{qty_contracts}",
            f"R$ {notional:,.2f}",
            f"R$ {total_credit:,.2f}"
        ]
    }
    df_results = pd.DataFrame(results_data)
    st.table(df_results)
    st.caption(f"Cálculo baseado no strike R$ {selected_strike:.2f} e vencimento {expiry.strftime('%Y-%m-%d')}")

elif asset_price <= 0:
    st.info("Aguardando preço do ativo para calcular...")
else:
    st.warning("Insira o preço da opção para calcular.")
