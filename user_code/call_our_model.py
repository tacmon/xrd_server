"""
XRD 自有模型 API 调用脚本

功能：递归读取指定文件夹中的 .txt 文件，逐条调用 XRD API 进行预测，
      将结果汇总保存为 JSON 文件。

用法：
    python call_our_model.py --folder ./final_data --output our_model_results.json
    python call_our_model.py --folder ./final_data --api http://localhost:8000/predict
"""

import os
import json
import requests
import argparse


def call_predict_api(content: str, api_url: str) -> dict:
    """
    调用 /predict API，发送文本数据并获取预测结果。

    Args:
        content: XRD 数据文本内容（2theta\\tintensity 格式）
        api_url: API 端点地址

    Returns:
        API 返回的 JSON 响应
    """
    payload = {"content": content}
    response = requests.post(api_url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="XRD 自有模型 API 调用脚本")
    parser.add_argument("--folder", type=str, default="./final_data",
                        help="存放待测 XRD .txt 文件的目录（支持子目录递归读取）")
    parser.add_argument("--api", type=str, default=None,
                        help="API 预测地址（默认从 .env 读取，若无则 http://localhost:8000/predict）")
    parser.add_argument("--output", type=str, default="our_model_results.json",
                        help="结果保存路径")
    args = parser.parse_args()

    # 确定 API 地址：优先命令行参数 > .env 文件 > 默认值
    api_url = args.api
    if api_url is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_URL="):
                        api_url = line.split("=", 1)[1].strip()
                        break
        if api_url is None:
            api_url = "http://localhost:8000/predict"

    spectra_dir = args.folder
    output_json = args.output

    if not os.path.exists(spectra_dir):
        print(f"Error: Directory '{spectra_dir}' not found.")
        return

    # 递归收集所有 .txt 文件
    files = []
    for root, dirs, filenames in os.walk(spectra_dir):
        for fname in sorted(filenames):
            if fname.endswith('.txt'):
                full_path = os.path.join(root, fname)
                # 使用相对路径作为 key，保留子目录结构信息
                rel_path = os.path.relpath(full_path, spectra_dir)
                files.append((rel_path, full_path))

    files.sort(key=lambda x: x[0])

    if not files:
        print(f"No .txt files found in {spectra_dir}")
        return

    print(f"Found {len(files)} files to process.")
    print(f"API endpoint: {api_url}")
    print()

    results = {}

    for idx, (rel_path, full_path) in enumerate(files):
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        print(f"[{idx+1}/{len(files)}] Processing {rel_path}...", end=" ", flush=True)

        try:
            resp_json = call_predict_api(content, api_url)

            if resp_json.get("code") == 200 and resp_json.get("status") == "success":
                result = resp_json.get("data")
                results[rel_path] = result
                print(f"Done. -> {result}")
            else:
                print(f"Failed (API Error: {resp_json.get('message')})")
                results[rel_path] = "API Error"
        except Exception as e:
            print(f"Error: {e}")
            results[rel_path] = "Request Error"

    # 保存汇总结果为 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nAll results saved to {output_json}")


if __name__ == "__main__":
    main()
