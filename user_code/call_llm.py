"""
XRD 通用大模型 API 调用脚本

功能：递归读取指定文件夹中的 .txt 文件，逐条调用通用大模型 API 进行预测，
      判断样品是否含有 AlN 物相，将结果汇总保存为 JSON 文件。

用法：
    python call_llm.py --folder ./final_data --key YOUR_API_KEY
    python call_llm.py --folder ./final_data --model Qwen/Qwen3-VL-30B-A3B-Instruct
"""

import os
import json
import requests
import argparse
import sys
import time


def get_high_quality_prompt(sample_id, data_content):
    """
    构造高质量的专业级 Prompt。
    """
    system_prompt = (
        "您是一位资深的材料表征专家，专注于 X 射线衍射 (XRD) 分析领域。\n"
        "您的任务是根据提供的 XRD 光谱数据（2-theta 角度 vs 相对强度），识别样品中是否包含标签为 'AlN_216' 的物相（氮化铝）。\n\n"
        "【输入说明】\n"
        "我将提供清洗后的 XRD 数据采样点（2-theta vs 相对强度）。请分析这些数据以判断物相。\n\n"
        "【专业背景知识】\n"
        "AlN_216 (氮化铝) 属于六方晶系 (Wurtzite 结构)，空间群 P6₃mc (No. 186)。其 XRD 特征峰通常出现在以下 2-theta 位置（Cu K-alpha 辐射）：\n"
        "- 约 33.2° 对应 (100) 晶面。\n"
        "- 约 36.0° 对应 (002) 晶面，这是极强的特征峰。\n"
        "- 约 37.9° 对应 (101) 晶面，这是最强的特征峰。\n"
        "- 约 49.8° 对应 (102) 晶面。\n"
        "- 约 59.3° 对应 (110) 晶面。\n"
        "如果在 36.0° 和 37.9° 附近观察到显著的衍射峰，请判定为包含 AlN_216。\n\n"
        "【输出要求】\n"
        "1. 请直接分析数据，不要解释过程。\n"
        "2. 最终输出必须是一个严格的 JSON 对象。\n"
        "3. JSON 格式：{\"result\": true} 或 {\"result\": false}\n"
        "4. 其中 true 表示识别出 AlN_216，false 表示未识别出。"
    )

    user_content = f"### 数据编号: {sample_id}\n数据内容:\n{data_content}\n---"

    return system_prompt, user_content

def main():
    parser = argparse.ArgumentParser(description="XRD 通用大模型逐个调用工具")
    parser.add_argument("--folder", type=str, required=True,
                        help="包含光谱数据文件的文件夹路径（支持子目录递归，自动过滤cif/jip等格式）")
    parser.add_argument("--api", type=str, default="https://api.siliconflow.cn/v1/chat/completions",
                        help="API URL")
    parser.add_argument("--key", type=str, default="记得更换为自己的api", help="API Key")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="模型名称")
    parser.add_argument("--output", type=str, default="llm_output_results.json",
                        help="输出的 JSON 文件名")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="处理每个文件后的延迟时间（秒），防止 API 频率限制")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"错误: 文件夹 {args.folder} 不存在。")
        sys.exit(1)

    # 递归收集所有数据文件
    txt_files = []
    ignore_extensions = ('.cif', '.jip', '.jpg', '.png', '.jpeg', '.json', '.md', '.py', '.sh', '.csv', '.zip', '.tar', '.gz')
    for root, dirs, filenames in os.walk(args.folder):
        for fname in sorted(filenames):
            # 排除以 . 开头的文件（如 .gitkeep, .DS_Store）以及特定的非光谱后缀
            if not fname.startswith('.') and not fname.lower().endswith(ignore_extensions):
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, args.folder)
                txt_files.append((rel_path, full_path))

    txt_files.sort(key=lambda x: x[0])

    if not txt_files:
        print("在指定文件夹中未找到待测数据文件。")
        sys.exit(1)
        
    print(f"共发现 {len(txt_files)} 个文件。正在逐个处理（API: {args.api}, Model: {args.model}）...")
    
    results = {}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.key}"
    }

    for idx, (rel_path, full_path) in enumerate(txt_files):
        print(f"[{idx+1}/{len(txt_files)}] Processing {rel_path}...", end=" ", flush=True)
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 过滤掉注释、表头信息，严防大模型作弊（泄露文件名/样本名）
                valid_lines = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # 只有真正能被转换为浮点数的两列，才是有效的光谱数据
                            float(parts[0])
                            float(parts[1])
                            valid_lines.append(f"{parts[0]}\t{parts[1]}")
                        except ValueError:
                            continue
                
                if not valid_lines:
                    print("Failed: No valid numeric data found.")
                    results[rel_path] = "Parsing Error"
                    continue
                
                # 采样逻辑：XRD数据通常点数极多（如8500个点），直接发送会超出大模型上下文或极其昂贵。
                # 采样间隔为 3，保留约 1/3 的数据点，足以保留峰形特征
                sampled_lines = valid_lines[::3]
                data_content = "\n".join(sampled_lines)
                
        except Exception as e:
            print(f"Failed to read: {e}")
            results[rel_path] = "Read Error"
            continue
        
        if not data_content:
            print("Skipped (Empty data).")
            results[rel_path] = False
            continue

        sys_prompt, user_prompt = get_high_quality_prompt(rel_path, data_content)
        
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        max_retries = 5
        retry_delay = 5  # 初始等待秒数
        
        for attempt in range(max_retries):
            try:
                response = requests.post(args.api, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 429:
                    print(f"Rate limited (429). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})", end="\r", flush=True)
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    continue
                    
                response.raise_for_status()
                
                result_data = response.json()
                raw_answer = result_data['choices'][0]['message']['content'].strip()
                
                # 提取 JSON 内容
                if "```json" in raw_answer:
                    raw_answer = raw_answer.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_answer:
                    raw_answer = raw_answer.split("```")[1].split("```")[0].strip()
                
                try:
                    id_results = json.loads(raw_answer)
                    if isinstance(id_results, dict) and "result" in id_results:
                        results[rel_path] = id_results["result"]
                    else:
                        results[rel_path] = id_results
                    print("Done.                                         ") # 清除重试行的残留文字
                    break # 成功则跳出重试循环
                except json.JSONDecodeError:
                    print("Error (JSON Parse Failed).")
                    results[rel_path] = "Parse Error"
                    break
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts ({e})")
                    results[rel_path] = "API Error"
                else:
                    time.sleep(retry_delay)
                    retry_delay *= 2

        # 暂停一段时间，避免触发频率限制
        if idx < len(txt_files) - 1:
            time.sleep(args.delay)

    # 保存最终结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n处理完成！结果已保存至 {args.output}")

if __name__ == "__main__":
    main()
