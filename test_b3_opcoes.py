# -*- coding: utf-8 -*-
"""
Teste completo da API B3 rapinegocios para opcoes
"""
import requests
import sys
import zipfile
import io
import pandas as pd
from datetime import date, timedelta

sys.stdout.reconfigure(encoding='utf-8')

def fetch_option_trades(ticker, trade_date=None):
    """
    Busca negocios de uma opcao especifica na B3
    
    Args:
        ticker: codigo da opcao (ex: BOVAN159)
        trade_date: data no formato YYYY-MM-DD (default: ontem)
    """
    if trade_date is None:
        # Usa o ultimo dia util (ontem ou sexta se for segunda)
        today = date.today()
        if today.weekday() == 0:  # Segunda
            trade_date = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        elif today.weekday() == 6:  # Domingo
            trade_date = (today - timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            trade_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    
    url = f"https://arquivos.b3.com.br/rapinegocios/tickercsv/{ticker}/{trade_date}"
    
    print(f"Buscando: {ticker} em {trade_date}")
    print(f"URL: {url}")
    
    try:
        r = requests.get(url, timeout=15)
        print(f"Status: {r.status_code}")
        
        if r.status_code != 200:
            print(f"Erro: {r.text[:200]}")
            return None
            
        # E um arquivo ZIP
        print(f"Content-Type: {r.headers.get('Content-Type', 'N/A')}")
        print(f"Tamanho: {len(r.content)} bytes")
        
        # Extrai o ZIP
        z = zipfile.ZipFile(io.BytesIO(r.content))
        files = z.namelist()
        print(f"\nArquivos no ZIP: {files}")
        
        # Le o primeiro arquivo (geralmente .txt ou .csv)
        for fname in files:
            print(f"\n=== Conteudo de {fname} ===")
            with z.open(fname) as f:
                content = f.read().decode('latin-1')
                
                # Mostra as primeiras linhas
                lines = content.split('\n')
                print(f"Total de linhas: {len(lines)}")
                print(f"\nPrimeiras 10 linhas:")
                for i, line in enumerate(lines[:10]):
                    print(f"  {i}: {line[:150]}")
                    
                # Tenta converter para DataFrame
                try:
                    # Detecta separador
                    if ';' in lines[0]:
                        sep = ';'
                    elif ',' in lines[0]:
                        sep = ','
                    else:
                        sep = '\t'
                    
                    df = pd.read_csv(io.StringIO(content), sep=sep, decimal=',', encoding='latin-1')
                    print(f"\n=== DataFrame ===")
                    print(f"Shape: {df.shape}")
                    print(f"Colunas: {list(df.columns)}")
                    print(df.head(10))
                    
                    # Estatisticas basicas
                    if 'Preco' in df.columns or 'PrecoNegocio' in df.columns:
                        price_col = 'Preco' if 'Preco' in df.columns else 'PrecoNegocio'
                        print(f"\n=== Estatisticas de Preco ===")
                        print(f"  Min: {df[price_col].min()}")
                        print(f"  Max: {df[price_col].max()}")
                        print(f"  Media: {df[price_col].mean():.2f}")
                        print(f"  Ultimo: {df[price_col].iloc[-1]}")
                    
                    return df
                    
                except Exception as e:
                    print(f"Erro ao parsear CSV: {e}")
                    return None
                    
    except Exception as e:
        print(f"Erro: {e}")
        return None


# Lista algumas opcoes de BOVA11 para testar
def list_option_tickers():
    """
    Gera lista de possiveis tickers de opcoes PUT para BOVA11
    """
    # BOVA + letra do mes (M=Jan, N=Fev, O=Mar... para PUTs)
    # + strike
    # Meses PUT: M(Jan), N(Fev), O(Mar), P(Abr), Q(Mai), R(Jun)
    #            S(Jul), T(Ago), U(Set), V(Out), W(Nov), X(Dez)
    
    # Vamos tentar o proximo vencimento (Fevereiro = N)
    strikes = [155, 156, 157, 158, 159, 160, 161, 162, 163]
    month_letter = 'N'  # Fevereiro
    
    return [f"BOVA{month_letter}{s}" for s in strikes]


if __name__ == "__main__":
    print("="*60)
    print("TESTE API B3 RAPINEGOCIOS")
    print("="*60)
    
    # Testa o ticker que o usuario encontrou
    print("\n--- Teste 1: BOVAN159 (PUT BOVA11 strike 159 Fev) ---")
    df = fetch_option_trades("BOVAN159", "2026-01-03")  # Sexta-feira
    
    # Tenta outros strikes
    print("\n\n--- Teste 2: Outros strikes disponiveis ---")
    for ticker in ["BOVAN155", "BOVAN160", "BOVAN165"]:
        print(f"\n{ticker}:")
        df = fetch_option_trades(ticker, "2026-01-03")
        if df is not None:
            print(f"  [OK] Dados encontrados!")
        else:
            print(f"  [X] Sem dados")
