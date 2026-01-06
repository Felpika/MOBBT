
import pandas as pd
import requests
import io
import streamlit as st

@st.cache_data(ttl=3600*4)
def obter_dados_tesouro():
    """
    Baixa e trata os dados atualizados do Tesouro Direto.
    Retorna DataFrame com colunas corrigidas.
    """
    # Link direto para o arquivo CSV de "Taxas e Preços de Títulos Públicos"
    url = 'https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv'
    
    try:
        # Adicionando headers para simular um navegador e evitar bloqueios/timeouts
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30) # Timeout de 30 segundos
        response.raise_for_status()
        
        df = pd.read_csv(io.BytesIO(response.content), sep=';', decimal=',')
        df['Data Vencimento'] = pd.to_datetime(df['Data Vencimento'], format='%d/%m/%Y')
        df['Data Base'] = pd.to_datetime(df['Data Base'], format='%d/%m/%Y')
        df['Tipo Titulo'] = df['Tipo Titulo'].astype('category')
        return df
    except Exception as e:
        print(f"Erro ao baixar dados do Tesouro (Tentativa 1): {e}")
        return pd.DataFrame()
