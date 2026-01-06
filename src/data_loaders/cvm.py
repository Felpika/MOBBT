
import requests
import pandas as pd
import io
import zipfile
import streamlit as st

@st.cache_data(ttl=3600*8)
def baixar_e_extrair_zip_cvm(url, nome_csv_interno, show_error=True):
    """Baixa e extrai um CSV de um arquivo ZIP da CVM em memória."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open(nome_csv_interno) as f:
                return pd.read_csv(f, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
    except Exception as e:
        if show_error:
            st.error(f"Erro ao baixar dados da CVM: {e}")
        return None
