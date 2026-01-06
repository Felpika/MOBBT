# -*- coding: utf-8 -*-
"""
Teste simplificado para opcoes.net.br
"""

import requests
import pandas as pd
import sys

# Força encoding UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def test_opcoes_net_br(ticker="BOVA11"):
    """
    Testa opcoes.net.br
    """
    print(f"\n{'='*60}")
    print(f"Testando opcoes.net.br para {ticker}")
    print('='*60)
    
    try:
        url = f"https://opcoes.net.br/opcoes/bovespa/{ticker.upper()}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        print(f"   URL: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"   Status: {response.status_code}")
        print(f"   Tamanho: {len(response.content)} bytes")
        
        if response.status_code == 200:
            print("[OK] Conseguiu acessar opcoes.net.br\n")
            
            # Tenta parsear tabelas HTML
            try:
                tables = pd.read_html(response.content, decimal=',', thousands='.')
                if tables:
                    print(f"   Encontradas {len(tables)} tabelas!\n")
                    for i, t in enumerate(tables):
                        print(f"\n=== Tabela {i} ({t.shape[0]} linhas x {t.shape[1]} colunas) ===")
                        print(f"Colunas: {list(t.columns)}")
                        if len(t) > 0:
                            print(t.head(10).to_string())
                        print()
                    return True
                else:
                    print("   Nenhuma tabela HTML encontrada")
                    return False
                    
            except Exception as parse_error:
                print(f"   Erro ao parsear tabelas: {parse_error}")
                return False
        else:
            print(f"[ERRO] Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERRO] {e}")
        return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTE OPCOES.NET.BR")
    print("="*60)
    
    result = test_opcoes_net_br("BOVA11")
    
    print("\n" + "="*60)
    print("RESULTADO:")
    print("="*60)
    print(f"  opcoes.net.br: {'OK' if result else 'FALHOU'}")
