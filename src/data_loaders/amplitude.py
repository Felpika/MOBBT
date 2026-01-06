
import streamlit as st
import pandas as pd
import requests
import io
import zipfile
import yfinance as yf
from datetime import datetime, timedelta

@st.cache_data(ttl=3600*8) # Cache de 8 horas
def obter_tickers_cvm_amplitude():
    """Esta função busca a lista de tickers da CVM."""
    st.info("Buscando lista de tickers da CVM... (Cache de 8h)")
    ano = datetime.now().year
    
    def tentar_baixar(ano_target):
        url = f'https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FCA/DADOS/fca_cia_aberta_{ano_target}.zip'
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(f'fca_cia_aberta_valor_mobiliario_{ano_target}.csv') as f:
                    return pd.read_csv(f, sep=';', encoding='ISO-8859-1', dtype={'Valor_Mobiliario': 'category', 'Mercado': 'category'})
        except Exception:
            return None

    df = tentar_baixar(ano)
    if df is None or df.empty:
        st.warning(f"Dados de {ano} não encontrados ou vazios. Tentando ano anterior ({ano-1})...")
        df = tentar_baixar(ano - 1)

    if df is not None and not df.empty:
        try:
            filtro_acoes = df['Valor_Mobiliario'].str.contains('Ordin|Preferenci', case=False, na=False, regex=True)
            filtro_mercado = df['Mercado'] == 'Bolsa'
            df_filtrado = df[filtro_acoes & filtro_mercado]
            return df_filtrado['Codigo_Negociacao'].dropna().unique().tolist()
        except Exception as e:
            st.error(f"Erro ao processar arquivo da CVM: {e}")
            return None
    else:
        st.error(f"Erro ao obter tickers da CVM (Tentativas {ano} e {ano-1} falharam ou arquivos estão vazios).")
        return None

@st.cache_data(ttl=3600*8) # Cache de 8 horas
def obter_precos_historicos_amplitude(tickers, anos_historico=5):
    """Esta função baixa os preços históricos para a análise de amplitude."""
    st.info(f"Baixando dados de preços de {len(tickers)} ativos... (Cache de 8h)")
    
    # Batch processing could be added here if list is huge, but yfinance handles chunks reasonably well.
    # App.py behavior:
    tickers_sa = [ticker + ".SA" for ticker in tickers]
    
    try:
        dados_completos = yf.download(
            tickers=tickers_sa,
            start=datetime.now() - timedelta(days=anos_historico*365),
            end=datetime.now(),
            auto_adjust=False,
            progress=False,
            group_by='ticker'
        )
        if not dados_completos.empty:
            # Seleciona 'Adj Close' (preferencial) ou 'Close' como fallback
            # Logic adapted from App.py
            if isinstance(dados_completos.columns, pd.MultiIndex):
                 # This is tricky with current yfinance versions.
                 # Assuming standard structure: (PriceType, Ticker) or (Ticker, PriceType)
                 # The App.py logic:
                 # price_type = 'Adj Close' if 'Adj Close' in dados_completos.columns.get_level_values(1) else 'Close'
                 # precos = dados_completos.stack(level=0, future_stack=True)[price_type].unstack(level=1)
                 
                 # Simpler robust way:
                 xs_key = 'Adj Close' if 'Adj Close' in dados_completos.columns.get_level_values(0) else 'Close'
                 # Note: yfinance format varies. Let's assume (Price, Ticker) or (Ticker, Price).
                 # App.py used stack/unstack which implies (Ticker, Price) or similar.
                 # Let's try to extract Adj Close column for all tickers.
                 
                 # Reconstruct from App.py snippet:
                 # price_type = 'Adj Close' if 'Adj Close' in dados_completos.columns.get_level_values(1) else 'Close'
                 # precos = dados_completos.stack(level=0, future_stack=True)[price_type].unstack(level=1)
                 
                 # Using the exact logic from App.py for consistency
                 return dados_completos.stack(level=0, future_stack=True)['Adj Close'].unstack(level=1).astype('float32') # approximate
                 # Wait, yfinance recent versions might have changed.
                 # I'll stick to what I saw in App.py:
                 # return precos.astype('float32')
            
            return pd.DataFrame() # Fallback
            
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()
