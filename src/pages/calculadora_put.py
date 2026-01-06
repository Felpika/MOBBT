
import streamlit as st
import pandas as pd
import math
import numpy as np
from datetime import date
import plotly.graph_objects as go
from src.models.put_utils import (
    get_selic_annual, get_asset_price_putcalc, get_next_expiration, generate_put_ticker
)
from src.data_loaders.b3_api import fetch_option_price_b3
from src.models.black_scholes import black_scholes_put, implied_volatility, calculate_greeks

def render():
    st.header("Calculadora de Venda de PUT (Cash-Secured Put)")
    st.info(
        "Ferramenta para analisar a venda de PUTs cobertas (Cash-Secured Puts). "
        "Essa estratégia gera renda (prêmio) com o compromisso de comprar a ação se cair muito. "
        "Ideal para ser remunerado enquanto espera para comprar uma ação no preço que você deseja."
    )
    st.markdown("---")
    
    # 1. Parâmetros Gerais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Ativo Objeto")
        asset_ticker = st.text_input("Ticker da Ação", "VALE3", help="Ex: VALE3, PETR4").upper().strip()
        asset_price = get_asset_price_putcalc(asset_ticker)
        
        col_price, col_selic = st.columns(2)
        col_price.metric("Preço Atual", f"R$ {asset_price:.2f}")
        
        selic_annual = get_selic_annual()
        selic_monthly = ((1 + selic_annual/100)**(1/12) - 1) * 100
        col_selic.metric("Selic Anual", f"{selic_annual:.2f}%", f"{selic_monthly:.2f}% a.m.")

    with col2:
        st.markdown("### Capital Disponível")
        collateral = st.number_input("Colateral (R$)", value=50000.0, step=1000.0, help="Valor total disponível para garantir a operação (Renda Fixa).")
        st.metric("Poder de Compra (Quantidade Aprox.)", f"{int(collateral / asset_price if asset_price > 0 else 0)} ações")
        
    current_date = date.today()
    from src.models.put_utils import get_third_friday
    available_expirations = [get_third_friday(current_date.year, m) for m in range(current_date.month+1, current_date.month+4)]
    # Handle year rollover if needed, but get_third_friday simple implementation might not cover it.
    # Assuming helper handles it or we improve it here.
    # Let's trust the logic from helper or simple manual fix:
    # Actually get_third_friday assumes passed year/month is valid.
    # Refined expiration logic:
    from dateutil.relativedelta import relativedelta
    available_expirations = []
    for i in range(1, 4):
        future_date = current_date + relativedelta(months=i)
        available_expirations.append(get_third_friday(future_date.year, future_date.month))
    
    with col3:
        st.markdown("### Vencimento")
        expiry_options = {
            f"{exp.strftime('%d/%m/%Y')} ({(exp - current_date).days} dias)": exp 
            for exp in available_expirations
        }
        selected_expiry_label = st.selectbox(
            "Selecione o Vencimento",
            options=list(expiry_options.keys()),
            key="putcalc_expiry_select"
        )
        expiry = expiry_options[selected_expiry_label]
        days_to_exp = (expiry - current_date).days
        st.metric("Dias até Vencimento", f"{days_to_exp} dias")

    # Sugestões
    suggested_strike = round(asset_price, 0) if asset_price > 0 else 0.0
    suggested_ticker = generate_put_ticker(asset_ticker[:4], expiry, suggested_strike) if asset_price > 0 else ""
    
    if 'putcalc_last_ticker' not in st.session_state:
        st.session_state.putcalc_last_ticker = asset_ticker
    
    if st.session_state.putcalc_last_ticker != asset_ticker:
        st.session_state.putcalc_last_ticker = asset_ticker
        st.session_state.putcalc_strike_input = suggested_strike
        st.session_state.pop('last_option_ticker', None)
        st.session_state.pop('b3_fetched_price', None)
        st.session_state.pop('b3_data', None)
        st.rerun()

    st.markdown("---")
    st.subheader("Dados da Opção")
    
    c_op1, c_op2, c_op3 = st.columns(3)
    
    with c_op1:
        if 'putcalc_strike_input' not in st.session_state:
            st.session_state.putcalc_strike_input = suggested_strike
        selected_strike = st.number_input("Strike Selecionado", step=0.5, format="%.2f", key="putcalc_strike_input")
    
    with c_op2:
        actual_ticker = generate_put_ticker(asset_ticker[:4], expiry, selected_strike) if selected_strike > 0 else ""
        st.text_input("Código da Opção (Teórico)", value=actual_ticker, disabled=True)

    if actual_ticker and st.session_state.get('last_option_ticker') != actual_ticker:
        with st.spinner(f"Buscando {actual_ticker} na B3..."):
            b3_data = fetch_option_price_b3(actual_ticker)
            if b3_data:
                st.session_state['b3_fetched_price'] = b3_data['last_price']
                st.session_state['b3_data'] = b3_data
            else:
                st.session_state['b3_fetched_price'] = 0.0
                st.session_state['b3_data'] = None
            st.session_state['last_option_ticker'] = actual_ticker

    with c_op3:
        b3_price = st.session_state.get('b3_fetched_price', 0.0)
        if b3_price > 0: st.metric("Prêmio B3 (Último)", f"R$ {b3_price:.2f}")
        else: st.warning("Sem dados B3")
        option_price = st.number_input("Prêmio Manual (opcional)", value=b3_price, step=0.01, format="%.2f", key="putcalc_premium")
        if st.session_state.get('b3_data'):
            b3 = st.session_state['b3_data']
            st.caption(f"📊 {b3['date']}: {b3['trades']} negócios, Vol: {b3['volume']:,.0f}")

    if selected_strike > 0 and option_price > 0 and asset_price > 0:
        raw_qty = math.floor(collateral / selected_strike)
        qty_contracts = (raw_qty // 100) * 100
        notional = qty_contracts * selected_strike
        yield_pct = (option_price / asset_price) * 100 
        pct_cdi = (yield_pct / selic_monthly) * 100 if selic_monthly > 0 else 0
        total_credit = qty_contracts * option_price
        
        break_even = selected_strike - option_price
        break_even_pct = ((asset_price - break_even) / asset_price) * 100
        max_loss = (selected_strike - option_price) * qty_contracts
        yield_anual = ((1 + yield_pct/100) ** 12 - 1) * 100
        pct_cdi_anual = (yield_anual / selic_annual) * 100 if selic_annual > 0 else 0
        moneyness = ((selected_strike - asset_price) / asset_price) * 100
        moneyness_label = "ATM" if abs(moneyness) < 1 else ("OTM" if moneyness < 0 else "ITM")

        st.markdown("## 📊 Resultados Principais")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Yield Mensal", f"{yield_pct:.2f}%")
        m2.metric("% do CDI", f"{pct_cdi:.0f}%")
        m3.metric("Qtd Contratos", f"{qty_contracts}")
        m4.metric("Crédito Total", f"R$ {total_credit:,.2f}")
        
        st.markdown("### 📈 Projeção Anual")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Yield Anualizado", f"{yield_anual:.2f}%")
        a2.metric("% CDI Anual", f"{pct_cdi_anual:.0f}%")
        a3.metric("Crédito Anual Est.", f"R$ {total_credit * 12:,.2f}")
        a4.metric("Moneyness", f"{moneyness_label} ({moneyness:+.1f}%)")
        
        st.markdown("### ⚠️ Análise de Risco")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Break-Even", f"R$ {break_even:.2f}")
        r2.metric("Margem de Segurança", f"{break_even_pct:.1f}%")
        r3.metric("Máx. Prejuízo Teórico", f"R$ {max_loss:,.2f}")
        r4.metric("Colateral Usado", f"{(notional/collateral)*100:.1f}%")

        # Gregas
        st.markdown("### 📐 Gregas (Black-Scholes)")
        try:
            S, K, T, r = asset_price, selected_strike, max(days_to_exp / 365.0, 0.001), selic_annual / 100
            iv = implied_volatility(option_price, S, K, T, r)
            greeks = calculate_greeks(S, K, T, r, iv, option_type="put")
            
            theta_daily = greeks['theta'] / 365
            
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Delta", f"{greeks['delta']:.3f}")
            g2.metric("Prob. Exercício", f"{abs(greeks['delta'])*100:.1f}%")
            g3.metric("Theta ($/dia)", f"R$ {theta_daily:.3f}", delta=f"R$ {theta_daily * qty_contracts:.2f}/dia total")
            g4.metric("Vol. Implícita", f"{iv * 100:.1f}%")
        except Exception as e:
            st.warning(f"Erro ao calcular gregas: {e}")

        # Payoff Chart
        st.markdown("### 📉 Gráfico de Payoff")
        price_range = np.linspace(selected_strike * 0.85, selected_strike * 1.15, 100)
        payoff = np.where(price_range >= selected_strike, option_price * qty_contracts, (option_price - (selected_strike - price_range)) * qty_contracts)
        
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=price_range, y=payoff, mode='lines', line=dict(color='#00D4FF', width=3), name='P&L', fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.2)'))
        fig_payoff.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_payoff.add_vline(x=asset_price, line_dash="dot", line_color="#FFB302", annotation_text=f"Atual: R${asset_price:.2f}")
        fig_payoff.add_vline(x=break_even, line_dash="dot", line_color="#FF4B4B", annotation_text=f"BE: R${break_even:.2f}")
        fig_payoff.update_layout(title="Perfil de Lucro/Prejuízo", xaxis_title="Preço no Vencimento", yaxis_title="R$", template='brokeberg', height=400)
        st.plotly_chart(fig_payoff, use_container_width=True)

    elif asset_price <= 0:
        st.info("Aguardando preço do ativo...")
    else:
        st.warning("Insira o preço da opção.")
