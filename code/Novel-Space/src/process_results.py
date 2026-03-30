import os
import sys
import argparse

# Set working directory to project root for easy path access
# Get the absolute path of the directory containing this script (src/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (Novel-Space/)
base_dir = os.path.dirname(script_dir)
# Change the current working directory to Novel-Space/
os.chdir(base_dir)
# Add the project root to sys.path so autoXRD can be imported
root_dir = os.path.dirname(base_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

import pandas as pd
import numpy as np
import ast
import re

# 默认主要物质（与 References 中的 CIF 文件名前缀一致）
DEFAULT_MAIN_SUBSTANCES = ["AlN_216"]

def parse_list_string(s):
    """
    Parses a string representation of a list, handling 'np.float64(...)' and other quirks.
    """
    if pd.isna(s) or s == "":
        return []
    # Replace np.float64(value) with just value, handling potential spaces
    s_clean = re.sub(r'np\.float64\s*\(\s*(.*?)\s*\)', r'\1', s)
    try:
        # ast.literal_eval handles standard list/string/number representations
        return ast.literal_eval(s_clean)
    except (ValueError, SyntaxError):
        # Fallback for more complex cases if needed
        return []

def process_results(input_file, output_file, main_substances=None):
    """
    处理 result.csv，根据主要物质进行分类判断。

    Args:
        input_file: 输入 CSV 文件路径 (result.csv)
        output_file: 输出 CSV 文件路径 (processed_result.csv)
        main_substances: 主要物质列表，默认为 ["AlN_216"]
    """
    if main_substances is None:
        main_substances = set(DEFAULT_MAIN_SUBSTANCES)
    else:
        main_substances = set(main_substances)

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
        return

    df = pd.read_csv(input_file)
    
    # 1. 解析所有预测结果
    parsed_phases = []
    parsed_confidences = []
    
    for idx, row in df.iterrows():
        phases = parse_list_string(row['Predicted phases'])
        confidences = parse_list_string(row['Confidence'])
        
        parsed_phases.append(phases)
        parsed_confidences.append(confidences)

    print(f"Processing with Main Substances: {sorted(list(main_substances))}")
    
    # 2. 处理数据：最大置信度且置信度过半
    processed_data = []
    for i in range(len(df)):
        phases = parsed_phases[i]
        confidences = parsed_confidences[i]
        
        # Default as Unidentified
        final_phase = "未识别"
        final_confidence = ""
        
        if phases:
            # Find the substance with the absolute highest confidence in this row
            max_conf = -1
            max_phase = None
            
            for p, c in zip(phases, confidences):
                if c > max_conf:
                    max_conf = c
                    max_phase = p
            
            # Check if this "winner" is one of the main substances and > 50%
            if max_phase in main_substances and max_conf > 50:
                final_phase = max_phase
                final_confidence = max_conf
        
        processed_data.append({
            'Filename': df.iloc[i]['Filename'],
            'Predicted phases': final_phase,
            'Confidence': final_confidence
        })
    
    output_df = pd.DataFrame(processed_data)
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 XRD 分析结果")
    parser.add_argument("--input", type=str, default="result.csv", help="输入 CSV 文件")
    parser.add_argument("--output", type=str, default="processed_result.csv", help="输出 CSV 文件")
    parser.add_argument("--main_substances", type=str, nargs="+", default=DEFAULT_MAIN_SUBSTANCES,
                        help="主要物质标签列表（默认: AlN_216）")
    args = parser.parse_args()
    
    process_results(args.input, args.output, args.main_substances)
