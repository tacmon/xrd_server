"""
XRD 预测结果评估脚本

功能：读取预测结果 JSON 文件，根据文件名中是否包含 AlN 作为真实标签，
      计算准确率、精确率、召回率和 F1 分数。

用法：
    python evaluate.py --file our_model_results.json
    python evaluate.py --file llm_output_results.json
"""

import json
import argparse
import os
import sys


def get_ground_truth(filename):
    """
    根据用户最新要求：完全满分等价于"文件名中包含 AlN 的，是 true；不包含 AlN 的是 false"。
    """
    return 'AlN' in filename


def calculate_metrics(results):
    tp = 0  # True Positive
    tn = 0  # True Negative
    fp = 0  # False Positive (第一类错误: 误报)
    fn = 0  # False Negative (第二类错误: 漏报)

    for filename, pred in results.items():
        truth = get_ground_truth(filename)
        # 确保预测值是布尔类型
        pred_bool = bool(pred)

        if truth and pred_bool:
            tp += 1
        elif not truth and not pred_bool:
            tn += 1
        elif not truth and pred_bool:
            fp += 1
        elif truth and not pred_bool:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results) if len(results) > 0 else 0

    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy
    }


def main():
    parser = argparse.ArgumentParser(description="XRD 预测结果 F1 分数打分工具")
    parser.add_argument("--file", type=str, required=True, help="待评估的预测结果 JSON 文件 (例如 output_results.json)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"错误: 找不到文件 {args.file}")
        sys.exit(1)

    with open(args.file, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print("错误: 无法解析 JSON 文件，请检查格式。")
            sys.exit(1)

    if not results:
        print("错误: JSON 文件内容为空。")
        sys.exit(1)

    metrics = calculate_metrics(results)

    print("\n" + "="*40)
    print(f"📊 XRD 模型预测性能评估报告")
    print("="*40)
    print(f"评估目标文件: {args.file}")
    print(f"样本数据总量: {len(results)}")
    print("-" * 40)
    print(f"✅ 真阳性 (TP): {metrics['TP']} (正确识别目标)")
    print(f"✅ 真阴性 (TN): {metrics['TN']} (正确排除非目标)")
    print(f"🚨 假阳性 (FP) [误报]: {metrics['FP']}")
    print(f"🚨 假阴性 (FN) [漏报]: {metrics['FN']}")
    print("-" * 40)
    print(f"🎯 准确率 (Accuracy) : {metrics['Accuracy']*100:6.2f}%")
    print(f"🎯 精确率 (Precision): {metrics['Precision']*100:6.2f}%")
    print(f"🎯 召回率 (Recall)   : {metrics['Recall']*100:6.2f}%")
    print(f"🔥 F1 分数 (F1-Score): {metrics['F1']*100:6.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
