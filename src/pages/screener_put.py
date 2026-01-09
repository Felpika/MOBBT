"""
Screener de PUT - Ações Líquidas
================================
Monitora opções PUT das ações mais líquidas com yields e indicadores fractais.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models.put_utils import (
    get_selic_annual, get_third_friday, generate_put_ticker,
    get_asset_price_yesterday
)
from src.data_loaders.b3_api import fetch_option_price_b3
from src.models.black_scholes import implied_volatility, calculate_greeks
from src.models.fractal_analytics import (
    calculate_hurst_exponent, get_hurst_interpretation,
    prob_exercise_bs, prob_exercise_fractal, calculate_historical_volatility,
    check_trend_filters, get_recommendation, calculate_iv_rank
)


# Lista de ações mais líquidas
LIQUID_TICKERS = [
    "VALE3", "PETR4", "MGLU3", "BBAS3", "BRAV3", "ITUB4", "B3SA3", 
    "BBDC4", "PRIO3", "RECV3", "ABEV3", "BPAC11", "ITSA4", "WEGE3", 
    "SUZB3", "BBSE3", "RAIZ4", "HAPV3", "CSAN3", "USIM5", "CSNA3",
    "GGBR4", "RENT3", "RAIL3", "BEEF3"
]


def get_atm_strike(spot_price: float) -> float:
    """Retorna strike ATM arredondado."""
    if spot_price < 20:
        return round(spot_price * 2) / 2  # Arredonda para 0.50
    elif spot_price < 50:
        return round(spot_price)  # Arredonda para 1.00
    else:
        return round(spot_price / 2) * 2  # Arredonda para 2.00


def scan_single_ticker(ticker: str, expiry: date, selic_annual: float) -> dict:
    """
    Escaneia um ticker individual e retorna métricas.
    
    Returns:
        dict com todas as métricas ou None se erro
    """
    try:
        # 1. Busca preço do ativo
        spot = get_asset_price_yesterday(ticker)
        if spot <= 0:
            return None
        
        # 2. Determina strike ATM
        strike = get_atm_strike(spot)
        
        # 3. Gera ticker da opção
        asset_code = ticker[:4]
        option_ticker = generate_put_ticker(asset_code, expiry, strike)
        
        # 4. Busca preço da opção na B3
        b3_data = fetch_option_price_b3(option_ticker)
        if not b3_data or b3_data['last_price'] <= 0:
            return None
        
        premium = b3_data['last_price']
        
        # 5. Calcula yields
        days_to_exp = (expiry - date.today()).days
        yield_period = (premium / spot) * 100
        yield_annual = ((1 + yield_period/100) ** (365/max(days_to_exp, 1)) - 1) * 100
        selic_period = ((1 + selic_annual/100) ** (days_to_exp/365) - 1) * 100
        pct_cdi = (yield_period / selic_period * 100) if selic_period > 0 else 0
        
        # 6. Moneyness
        moneyness = ((strike - spot) / spot) * 100
        
        # 7. Busca histórico para análise fractal
        full_ticker = f"{ticker}.SA"
        start_date = date.today() - timedelta(days=int(252 * 1.5))
        hist = yf.download(full_ticker, start=start_date.strftime('%Y-%m-%d'), 
                          end=date.today().strftime('%Y-%m-%d'), progress=False)
        
        if hist.empty or len(hist) < 50:
            return None
        
        # Extrai série de preços
        if 'Adj Close' in hist.columns:
            close_prices = hist['Adj Close'].tail(252)
        elif 'Close' in hist.columns:
            close_prices = hist['Close'].tail(252)
        else:
            close_prices = hist.iloc[:, 0].tail(252)
        
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze()
        
        # 8. Calcula Hurst e volatilidade
        hurst = calculate_hurst_exponent(close_prices)
        hist_vol = calculate_historical_volatility(close_prices)
        
        # 9. Calcula probabilidades
        T = max(days_to_exp / 365.0, 0.001)
        r = selic_annual / 100
        
        # Tenta calcular IV, usa volatilidade histórica como fallback
        try:
            iv = implied_volatility(premium, spot, strike, T, r)
            if iv <= 0:
                iv = hist_vol
        except:
            iv = hist_vol
        
        prob_bs = prob_exercise_bs(spot, strike, T, r, iv)
        prob_frac = prob_exercise_fractal(spot, strike, T, r, iv, hurst)
        
        # 10. Filtros de tendência
        filters = check_trend_filters(close_prices)
        
        # 11. Interpretação Hurst
        recent_return = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) * 100 if len(close_prices) >= 20 else 0
        interpretation, trend_dir, _ = get_hurst_interpretation(hurst, recent_return)
        
        # 12. Recomendação
        classification, rec_text, risk_level, rec_color = get_recommendation(hurst, filters, spot)
        
        # 13. IV Rank
        iv_rank_data = calculate_iv_rank(iv, close_prices)
        iv_rank = iv_rank_data['iv_rank']
        iv_signal = iv_rank_data['sell_signal']
        
        return {
            'Ticker': ticker,
            'Opção': option_ticker,
            'Spot': round(spot, 2),
            'Strike': round(strike, 2),
            'Prêmio': round(premium, 2),
            'Moneyness': round(moneyness, 1),
            'Yield %': round(yield_period, 2),
            'Yield A.': round(yield_annual, 1),
            '% CDI': round(pct_cdi, 0),
            'Hurst': round(hurst, 3),
            'Tipo': interpretation,
            'Direção': trend_dir,
            'SMA21': '✅' if filters['filter_a'] else '❌',
            'Mom 30d': '✅' if filters['filter_b'] else '❌',
            'Slope': '✅' if filters['filter_c'] else '❌',
            'Prob BS': round(prob_bs * 100, 1),
            'Prob Frac': round(prob_frac * 100, 1),
            'Recomendação': classification,
            'Risco': risk_level,
            '_rec_color': rec_color,
            'Vol': round(b3_data.get('volume', 0), 0),
            'IV Rank': round(iv_rank, 0),
            'IV Sinal': iv_signal,
        }
        
    except Exception as e:
        return None


def run_full_scan(tickers: list, expiry: date, progress_callback=None) -> pd.DataFrame:
    """
    Escaneia todos os tickers em paralelo.
    
    Returns:
        DataFrame com resultados
    """
    selic_annual = get_selic_annual()
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(scan_single_ticker, ticker, expiry, selic_annual): ticker 
            for ticker in tickers
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if progress_callback:
                progress_callback(completed / len(tickers))
            
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Ordena por Yield decrescente
    df = df.sort_values('Yield %', ascending=False)
    
    return df


def apply_recommendation_style(val):
    """Aplica cor de fundo baseada na recomendação."""
    colors = {
        'VENDA FORTE': 'background-color: rgba(57, 229, 140, 0.3)',
        'OPORTUNIDADE': 'background-color: rgba(0, 212, 255, 0.3)',
        'NEUTRO': 'background-color: rgba(99, 110, 250, 0.3)',
        'CAUTELA': 'background-color: rgba(255, 179, 2, 0.3)',
        'RISCO ALTO': 'background-color: rgba(255, 75, 75, 0.3)',
    }
    return colors.get(val, '')


def apply_hurst_style(val):
    """Aplica cor baseada no Hurst."""
    try:
        h = float(val)
        if h > 0.55:
            return 'color: #39E58C'  # Verde - persistente
        elif h < 0.45:
            return 'color: #FFB302'  # Amarelo - reversão
        else:
            return 'color: #636EFA'  # Azul - random
    except:
        return ''


def render():
    st.header("📊 Screener de PUT - Ações Líquidas")
    st.info(
        "Monitora opções PUT das ações mais líquidas da B3 com análise de yields, "
        "indicadores fractais e recomendações de venda."
    )
    
    # Controles
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Seleção de vencimento
        current_date = date.today()
        available_expirations = []
        for i in range(1, 4):
            future_date = current_date + relativedelta(months=i)
            available_expirations.append(get_third_friday(future_date.year, future_date.month))
        
        expiry_options = {
            f"{exp.strftime('%d/%m/%Y')} ({(exp - current_date).days} dias)": exp 
            for exp in available_expirations
        }
        selected_expiry_label = st.selectbox(
            "Vencimento",
            options=list(expiry_options.keys()),
            key="screener_expiry"
        )
        expiry = expiry_options[selected_expiry_label]
    
    with col2:
        # Filtro de recomendação
        rec_filter = st.multiselect(
            "Filtrar por Recomendação",
            options=["VENDA FORTE", "OPORTUNIDADE", "NEUTRO", "CAUTELA", "RISCO ALTO"],
            default=["VENDA FORTE", "OPORTUNIDADE", "NEUTRO"],
            key="screener_rec_filter"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_button = st.button("🔍 Escanear", type="primary", use_container_width=True)
    
    # Métricas gerais
    selic = get_selic_annual()
    days_to_exp = (expiry - current_date).days
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selic Anual", f"{selic:.2f}%")
    m2.metric("Dias até Vencimento", f"{days_to_exp}")
    m3.metric("Ativos Monitorados", f"{len(LIQUID_TICKERS)}")
    m4.metric("Vencimento", expiry.strftime("%d/%m/%Y"))
    
    st.markdown("---")
    
    # Execução do scan
    if scan_button or 'screener_df' in st.session_state:
        if scan_button:
            progress_bar = st.progress(0, text="Escaneando ativos...")
            
            def update_progress(pct):
                progress_bar.progress(pct, text=f"Escaneando ativos... {int(pct*100)}%")
            
            df = run_full_scan(LIQUID_TICKERS, expiry, update_progress)
            progress_bar.empty()
            
            if df.empty:
                st.warning("Nenhuma opção encontrada com dados válidos.")
                return
            
            st.session_state['screener_df'] = df
            st.session_state['screener_expiry_used'] = expiry
        
        df = st.session_state.get('screener_df', pd.DataFrame())
        
        if df.empty:
            st.warning("Nenhum dado disponível. Clique em 'Escanear' para buscar.")
            return
        
        # Aplica filtro de recomendação
        if rec_filter:
            df_filtered = df[df['Recomendação'].isin(rec_filter)]
        else:
            df_filtered = df
        
        # Estatísticas resumidas
        st.markdown("### 📈 Resumo do Scan")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Opções Encontradas", f"{len(df)}")
        s2.metric("Yield Médio", f"{df['Yield %'].mean():.2f}%")
        s3.metric("Yield Máximo", f"{df['Yield %'].max():.2f}%")
        
        venda_forte_count = len(df[df['Recomendação'] == 'VENDA FORTE'])
        s4.metric("VENDA FORTE", f"{venda_forte_count}")
        
        risco_alto_count = len(df[df['Recomendação'] == 'RISCO ALTO'])
        s5.metric("RISCO ALTO", f"{risco_alto_count}")
        
        st.markdown("---")
        
        # Tabela principal
        st.markdown("### 📋 Resultados do Screener")
        
        # Colunas para exibição
        display_cols = [
            'Ticker', 'Opção', 'Spot', 'Strike', 'Prêmio', 'Moneyness',
            'Yield %', '% CDI', 'IV Rank', 'IV Sinal', 'Hurst', 'Tipo', 'Direção',
            'SMA21', 'Mom 30d', 'Slope', 'Prob BS', 'Prob Frac', 'Recomendação'
        ]
        
        df_display = df_filtered[display_cols].copy()
        
        # Estilização
        styled_df = df_display.style.applymap(
            apply_recommendation_style, subset=['Recomendação']
        ).applymap(
            apply_hurst_style, subset=['Hurst']
        ).format({
            'Spot': 'R$ {:.2f}',
            'Strike': 'R$ {:.2f}',
            'Prêmio': 'R$ {:.2f}',
            'Moneyness': '{:.1f}%',
            'Yield %': '{:.2f}%',
            '% CDI': '{:.0f}%',
            'IV Rank': '{:.0f}%',
            'Hurst': '{:.3f}',
            'Prob BS': '{:.1f}%',
            'Prob Frac': '{:.1f}%',
        })
        
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Legenda
        with st.expander("📖 Legenda"):
            leg1, leg2 = st.columns(2)
            
            with leg1:
                st.markdown("""
                **Indicadores de Direção:**
                - **SMA21**: ✅ se Preço > SMA 21 dias
                - **Mom 30d**: ✅ se Momentum 30 dias > 0
                - **Slope**: ✅ se Inclinação da regressão linear > 0
                
                **Interpretação do Hurst:**
                - 🟢 **> 0.55**: Persistente (tendência continua)
                - 🔵 **0.45-0.55**: Random Walk
                - 🟡 **< 0.45**: Reversão à média
                """)
            
            with leg2:
                st.markdown("""
                **Recomendações:**
                - 🟢 **VENDA FORTE**: Alta persistente + todos filtros bullish
                - 🔵 **OPORTUNIDADE**: Ativo esticado para baixo com reversão
                - 🟣 **NEUTRO**: Siga as probabilidades Black-Scholes
                - 🟡 **CAUTELA**: Exaustão de topo
                - 🔴 **RISCO ALTO**: Queda persistente sem suporte
                """)
        
        # Download CSV
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"screener_put_{date.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
