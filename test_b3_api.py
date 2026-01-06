# -*- coding: utf-8 -*-
"""
Teste para buscar dados de opções da B3
"""

import requests
import pandas as pd
import sys
import json

# Força encoding UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def test_b3_cotacoes():
    """
    Testa acesso ao site de cotações da B3
    """
    print(f"\n{'='*60}")
    print("Testando B3 Cotacoes")
    print('='*60)
    
    try:
        # Página principal de cotações
        url = "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/cotacoes/"
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
            print("[OK] Conseguiu acessar B3 Cotacoes\n")
            
            # Procura por arquivos de cotação ou APIs
            content = response.text
            
            # Verifica se há links para arquivos
            if 'COTAHIST' in content:
                print("   Encontrado referencia a COTAHIST")
            if 'SerHist' in content:
                print("   Encontrado referencia a SerHist")
            if '.zip' in content.lower():
                print("   Encontrado referencia a arquivos .zip")
            if 'api' in content.lower():
                print("   Encontrado referencia a API")
                
            # Tenta parsear tabelas HTML
            try:
                tables = pd.read_html(response.content, decimal=',', thousands='.')
                if tables:
                    print(f"\n   Encontradas {len(tables)} tabelas HTML!")
                    for i, t in enumerate(tables):
                        print(f"\n=== Tabela {i} ({t.shape[0]} linhas x {t.shape[1]} colunas) ===")
                        print(f"Colunas: {list(t.columns)}")
                        if len(t) > 0:
                            print(t.head(5).to_string())
            except Exception as e:
                print(f"   Sem tabelas HTML parseáveis: {e}")
                
            return True
        else:
            print(f"[ERRO] Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[ERRO] {e}")
        return False


def test_b3_api_opcoes():
    """
    Testa a API interna da B3 para opções
    """
    print(f"\n{'='*60}")
    print("Testando B3 API de Opcoes")
    print('='*60)
    
    # A B3 usa uma API interna para algumas consultas
    # Vamos tentar encontrar endpoints conhecidos
    
    apis = [
        # API de índices (funciona bem)
        "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJpbmRleCI6IkJPVkEiLCJsYW5ndWFnZSI6InB0LWJyIn0=",
        # API de derivativos (pode ter opções)
        "https://www.b3.com.br/pt_br/json/cotacoes/opcoes.json",
        # API de opções BOVA
        "https://opcoes.net.br/listaopcoes/completa?idAcao=BOVA11&liession=all",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
    }
    
    for url in apis:
        print(f"\n   Testando: {url[:60]}...")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                print(f"   Content-Type: {content_type}")
                
                if 'json' in content_type.lower():
                    try:
                        data = response.json()
                        if isinstance(data, dict):
                            print(f"   JSON keys: {list(data.keys())[:10]}")
                            # Mostra preview dos dados
                            for k, v in list(data.items())[:3]:
                                if isinstance(v, list) and len(v) > 0:
                                    print(f"   {k}: lista com {len(v)} items")
                                    if isinstance(v[0], dict):
                                        print(f"       Keys do primeiro item: {list(v[0].keys())[:5]}")
                                else:
                                    print(f"   {k}: {str(v)[:50]}")
                        elif isinstance(data, list):
                            print(f"   JSON: lista com {len(data)} items")
                            if len(data) > 0 and isinstance(data[0], dict):
                                print(f"   Keys: {list(data[0].keys())[:10]}")
                    except:
                        print(f"   Resposta (primeiros 200 chars): {response.text[:200]}")
                else:
                    print(f"   Resposta HTML/Text (primeiros 200 chars): {response.text[:200]}")
        except Exception as e:
            print(f"   Erro: {e}")
    
    return True


def test_opcoes_net_api():
    """
    Testa APIs específicas do opcoes.net.br
    """
    print(f"\n{'='*60}")
    print("Testando APIs opcoes.net.br")
    print('='*60)
    
    # O site provavelmente usa AJAX para carregar dados
    apis = [
        "https://opcoes.net.br/listaopcoes/completa?idAcao=BOVA11&liession=all",
        "https://opcoes.net.br/opcoes/api/opcoes/BOVA11",
        "https://opcoes.net.br/api/opcoes/bovespa/BOVA11",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'X-Requested-With': 'XMLHttpRequest',  # Simula AJAX
    }
    
    for url in apis:
        print(f"\n   Testando: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                print(f"   Content-Type: {content_type}")
                print(f"   Tamanho: {len(response.content)} bytes")
                
                # Tenta parsear como JSON
                try:
                    data = response.json()
                    print(f"   [JSON VALIDO!]")
                    if isinstance(data, dict):
                        print(f"   Keys: {list(data.keys())}")
                        # Mostra estrutura
                        for k, v in data.items():
                            if isinstance(v, list):
                                print(f"   {k}: lista com {len(v)} items")
                                if len(v) > 0:
                                    print(f"       Primeiro item: {str(v[0])[:100]}")
                            else:
                                print(f"   {k}: {str(v)[:80]}")
                    elif isinstance(data, list):
                        print(f"   Lista com {len(data)} items")
                        if len(data) > 0:
                            print(f"   Primeiro: {str(data[0])[:100]}")
                    return True
                except:
                    # Não é JSON, mostra HTML
                    print(f"   Nao e JSON. Conteudo: {response.text[:300]}")
        except Exception as e:
            print(f"   Erro: {e}")
    
    return False


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTE DE APIS B3 E OPCOES")
    print("="*60)
    
    results = {
        "b3_cotacoes": test_b3_cotacoes(),
        "b3_api_opcoes": test_b3_api_opcoes(),
        "opcoes_net_api": test_opcoes_net_api(),
    }
    
    print("\n" + "="*60)
    print("RESUMO:")
    print("="*60)
    for name, ok in results.items():
        print(f"  {name:20s}: {'OK' if ok else 'FALHOU'}")
