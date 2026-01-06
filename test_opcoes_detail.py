# -*- coding: utf-8 -*-
"""
Teste detalhado da API opcoes.net.br
"""

import requests
import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def fetch_options_data(ticker="BOVA11"):
    """
    Busca dados de opções do opcoes.net.br
    """
    url = f"https://opcoes.net.br/listaopcoes/completa?idAcao={ticker}&liession=all"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
    }
    
    print(f"Buscando opcoes para {ticker}...")
    print(f"URL: {url}\n")
    
    response = requests.get(url, headers=headers, timeout=15)
    
    if response.status_code != 200:
        print(f"Erro: Status {response.status_code}")
        return None
    
    data = response.json()
    
    print(f"Success: {data.get('success')}")
    print(f"\nKeys em 'data': {list(data.get('data', {}).keys())}")
    
    # Explora a estrutura
    data_obj = data.get('data', {})
    
    for key, value in data_obj.items():
        print(f"\n=== {key} ===")
        if isinstance(value, str):
            # Pode ser JSON embutido
            if value.startswith('[') or value.startswith('{'):
                try:
                    parsed = json.loads(value)
                    print(f"JSON embutido parseado!")
                    if isinstance(parsed, list):
                        print(f"Lista com {len(parsed)} items")
                        if len(parsed) > 0:
                            print(f"Primeiro item: {parsed[0]}")
                    else:
                        print(f"Dict com keys: {list(parsed.keys())[:10]}")
                except:
                    print(f"String (primeiros 200 chars): {value[:200]}")
            else:
                print(f"String: {value[:200] if len(value) > 200 else value}")
        elif isinstance(value, list):
            print(f"Lista com {len(value)} items")
            if len(value) > 0:
                print(f"Primeiro item:")
                first = value[0]
                if isinstance(first, dict):
                    for k, v in first.items():
                        print(f"  {k}: {v}")
                elif isinstance(first, list):
                    print(f"  Lista aninhada com {len(first)} elementos")
                    for i, item in enumerate(first[:5]):
                        print(f"    [{i}]: {item}")
                else:
                    print(f"  {first}")
                    
            # Mostra mais alguns
            if len(value) > 1:
                print(f"\nTotal de {len(value)} opcoes encontradas!")
                print("\nPrimeiras 5 opcoes:")
                for i, opt in enumerate(value[:5]):
                    print(f"\n  Opcao {i+1}:")
                    if isinstance(opt, list):
                        # Formato de lista
                        for j, val in enumerate(opt):
                            print(f"    [{j}]: {val}")
                    elif isinstance(opt, dict):
                        for k, v in opt.items():
                            print(f"    {k}: {v}")
        elif isinstance(value, dict):
            print(f"Dict com keys: {list(value.keys())}")
            for k, v in list(value.items())[:5]:
                print(f"  {k}: {v}")
        else:
            print(f"Valor: {value}")
    
    return data


if __name__ == "__main__":
    print("="*60)
    print("ANALISE DETALHADA DA API OPCOES.NET.BR")
    print("="*60)
    
    data = fetch_options_data("BOVA11")
    
    if data:
        print("\n" + "="*60)
        print("ESTRUTURA COMPLETA (JSON):")
        print("="*60)
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
