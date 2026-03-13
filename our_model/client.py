import os
import requests
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="XRD 模型调用脚本")
    parser.add_argument("--folder", type=str, default="./Spectra", help="存放待测 XRD .txt 文件的目录")
    parser.add_argument("--api", type=str, default="http://localhost:8000/predict", help="API 预测地址")
    parser.add_argument("--output", type=str, default="results.json", help="结果保存路径")
    args = parser.parse_args()

    spectra_dir = args.folder
    api_url = args.api
    output_json = args.output

    if not os.path.exists(spectra_dir):
        print(f"Error: Directory '{spectra_dir}' not found.")
        return

    results = {}
    files = sorted([f for f in os.listdir(spectra_dir) if f.endswith('.txt')])
    
    if not files:
        print(f"No .txt files found in {spectra_dir}")
        return

    print(f"Found {len(files)} files to process.")

    for filename in files:
        file_path = os.path.join(spectra_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Processing {filename}...", end=" ", flush=True)
        
        try:
            payload = {
                "content": content
            }
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                resp_json = response.json()
                if resp_json.get("code") == 200 and resp_json.get("status") == "success":
                    result = resp_json.get("data")
                    results[filename] = result
                    print("Done.")
                else:
                    print(f"Failed (API Error: {resp_json.get('message')})")
            else:
                print(f"Failed (HTTP Status: {response.status_code})")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")

    # 保存汇总结果为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nAll results saved to {output_json}")

if __name__ == "__main__":
    main()
