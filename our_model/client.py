import os
import requests
import json

# API 配置
API_URL = "http://localhost:8000/predict"
SPECTRA_DIR = "./Spectra"  # 本地存放待测文件的目录
OUTPUT_JSON = "results.json"

def main():
    if not os.path.exists(SPECTRA_DIR):
        print(f"Error: Directory '{SPECTRA_DIR}' not found.")
        return

    results = {}
    files = sorted([f for f in os.listdir(SPECTRA_DIR) if f.endswith('.txt')])
    
    if not files:
        print(f"No .txt files found in {SPECTRA_DIR}")
        return

    print(f"Found {len(files)} files to process.")

    for filename in files:
        file_path = os.path.join(SPECTRA_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Processing {filename}...", end=" ", flush=True)
        
        try:
            payload = {
                "filename": filename,
                "content": content
            }
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                results[filename] = result
                print("Done.")
            else:
                print(f"Failed (Status: {response.status_code})")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    # 保存汇总结果为 JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nAll results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
