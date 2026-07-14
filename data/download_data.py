import requests
import os

def download_eu_ai_act():
    # Direct link to the final official EU AI Act text (Regulation 2024/1689) on EUR-Lex
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689"
    output_path = "data/eu_ai_act.pdf"
    
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    
    print(f"📥 Downloading official EU AI Act from EUR-Lex...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Successfully downloaded to {output_path}")
    else:
        print(f"❌ Failed to download automatically (Status: {response.status_code}).")
        print("👉 Please download it manually from this link and save it as 'data/eu_ai_act.pdf':")
        print("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689")

if __name__ == "__main__":
    download_eu_ai_act()