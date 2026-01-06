
import streamlit as st
import yfinance as yf
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
        asset_ticker = st.text_input("Ticker da Ação", "", help="Ex: VALE3, PETR4").upper().strip()
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
            
            theta_daily = greeks['theta_daily']
            
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Delta", f"{greeks['delta']:.3f}")
            g2.metric("Prob. Exercício", f"{abs(greeks['delta'])*100:.1f}%")
            g3.metric("Theta ($/dia)", f"R$ {theta_daily:.3f}", delta=f"R$ {theta_daily * qty_contracts:.2f}/dia total")
            g4.metric("Vol. Implícita", f"{iv * 100:.1f}%")
        except Exception as e:
            st.warning(f"Erro ao calcular gregas: {e}")

        st.markdown("---")
        
        # === ANÁLISE DE PROBABILIDADE HISTÓRICA (RECUPERADO) ===
        st.markdown("### 📊 Probabilidade Histórica de Exercício")
        
        # Calcula dias até o vencimento
        days_to_expiry_hist = (expiry - current_date).days
        
        # Busca dados históricos do ativo
        with st.spinner(f"Analisando histórico de {asset_ticker}..."):
            try:
                full_ticker = asset_ticker if asset_ticker.endswith(".SA") else f"{asset_ticker}.SA"
                # Use a larger period to get enough samples
                hist_data = yf.download(full_ticker, period="10y", progress=False)
                
                if not hist_data.empty and len(hist_data) > days_to_expiry_hist:
                    # Calcula retornos para o período igual ao tempo até vencimento
                    if 'Adj Close' in hist_data.columns:
                        close_col = 'Adj Close'
                    elif 'Close' in hist_data.columns:
                        close_col = 'Close'
                    else:
                        close_col = hist_data.columns[0] # Fallback
                        
                    hist_data['Forward_Return'] = (hist_data[close_col].shift(-days_to_expiry_hist) / hist_data[close_col] - 1) * 100
                    
                    # Remove NaNs (últimos N dias não terão retorno forward)
                    returns = hist_data['Forward_Return'].dropna()
                    
                    # Conta quantas vezes caiu mais que a margem de segurança
                    threshold = -break_even_pct  # Negativo porque é queda
                    breaches = returns[returns < threshold]
                    total_periods = len(returns)
                    breach_count = len(breaches)
                    
                    if total_periods > 0:
                        probability = (breach_count / total_periods) * 100
                        
                        # Exibe resultados
                        p1, p2, p3, p4 = st.columns(4)
                        
                        # Cor da probabilidade baseada no risco
                        if probability < 5:
                            prob_color = "normal"
                        elif probability < 15:
                            prob_color = "off"
                        else:
                            prob_color = "inverse"
                        
                        p1.metric(
                            "Prob. Histórica de Exercício", 
                            f"{probability:.1f}%",
                            delta=f"{breach_count} vezes em {total_periods}",
                            delta_color=prob_color,
                            help=f"Em {total_periods} períodos de {days_to_expiry_hist} dias, o ativo caiu mais de {break_even_pct:.1f}% em {breach_count} vezes"
                        )
                        
                        p2.metric(
                            "Dias até Vencimento",
                            f"{days_to_expiry_hist}",
                            help="Período utilizado para análise histórica"
                        )
                        
                        # Pior queda histórica no período
                        worst_drop = returns.min()
                        p3.metric(
                            "Pior Queda no Período",
                            f"{worst_drop:.1f}%",
                            help=f"Maior queda histórica em {days_to_expiry_hist} dias"
                        )
                        
                        # Queda média quando há exercício
                        if breach_count > 0:
                            avg_breach = breaches.mean()
                            p4.metric(
                                "Queda Média (se exercido)",
                                f"{avg_breach:.1f}%",
                                help="Média das quedas quando ultrapassa o break-even"
                            )
                        else:
                            p4.metric(
                                "Queda Média (se exercido)",
                                "N/A",
                                help="Não houve exercício histórico com esses parâmetros"
                            )
                        
                        # Expander com detalhes
                        with st.expander("📈 Ver distribuição histórica de retornos"):
                            # Histograma dos retornos - versão simplificada
                            fig_hist = go.Figure()
                            
                            # Histograma único com intervalos de 1%
                            fig_hist.add_trace(go.Histogram(
                                x=returns,
                                xbins=dict(size=1),
                                name='Retornos',
                                marker_color='#00D4FF',
                                opacity=0.8
                            ))
                            
                            # Adiciona área sombreada para zona de exercício (esquerda do threshold)
                            # y_max estimation
                            counts, _ = np.histogram(returns, bins='auto')
                            y_max = counts.max() * 1.1 if len(counts) > 0 else 10

                            fig_hist.add_vrect(
                                x0=returns.min(),
                                x1=threshold,
                                fillcolor="rgba(255, 75, 75, 0.3)",
                                layer="below",
                                line_width=0,
                                annotation_text="Zona de Exercício",
                                annotation_position="top left",
                                annotation_font_color="#FF4B4B"
                            )
                            
                            # Linha vertical no threshold (break-even)
                            fig_hist.add_vline(
                                x=threshold, 
                                line_dash="solid", 
                                line_color="#FF4B4B",
                                line_width=2,
                                annotation_text=f"Break-Even: {threshold:.1f}%",
                                annotation_position="top right",
                                annotation_font_color="#FF4B4B"
                            )
                            
                            # Linha vertical no zero
                            fig_hist.add_vline(
                                x=0, 
                                line_dash="dash", 
                                line_color="#39E58C",
                                line_width=1,
                                annotation_text="0%",
                                annotation_position="bottom right"
                            )
                            
                            fig_hist.update_layout(
                                title=f"Distribuição de Retornos em {days_to_expiry_hist} dias ({total_periods} observações)",
                                xaxis_title="Retorno (%)",
                                yaxis_title="Frequência",
                                template='brokeberg',
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                            st.caption(f"Análise baseada em {total_periods} períodos de {days_to_expiry_hist} dias nos últimos 10 anos de dados disponíveis.")
                    else:
                        st.warning("Dados históricos insuficientes para análise.")
            except Exception as e:
                st.error(f"Erro ao analisar histórico: {e}")

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
