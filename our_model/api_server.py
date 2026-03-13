import os
import shutil
import subprocess
import csv
from fastapi import FastAPI, Body

app = FastAPI()

# 容器内的工作目录
WORKING_DIR = "/workspace"
SPECTRA_DIR = os.path.join(WORKING_DIR, "Spectra")
RESULT_CSV = os.path.join(WORKING_DIR, "result.csv")

# 确保 Spectra 目录存在
os.makedirs(SPECTRA_DIR, exist_ok=True)

def cleanup_spectra():
    """预测完成后清空 Spectra 目录"""
    if not os.path.exists(SPECTRA_DIR):
        return
    for filename in os.listdir(SPECTRA_DIR):
        file_path = os.path.join(SPECTRA_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def run_prediction():
    """调用 run_CNN.py 进行预测"""
    try:
        # 使用 subprocess 调用原有的脚本
        subprocess.run(["python", "run_CNN.py", "--inc_pdf"], check=True, cwd=WORKING_DIR)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Prediction failed with exit code {e.returncode}")
        return False

def parse_results(target_filename: str):
    """解析 result.csv 并生成要求的 JSON 格式"""
    if not os.path.exists(RESULT_CSV):
        return False

    is_positive = False
    with open(RESULT_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Filename']
            if filename != target_filename:
                continue

            preds = []
            try:
                # 解析字符串形式的列表，例如 "['CrSiTe3_148', 'SiTe2_164']"
                preds = ast.literal_eval(row['Predicted phases'])
            except:
                preds = row['Predicted phases']
            
            # 判断是否包含 CrSiTe3 (正例判断逻辑参考了评测脚本)
            if isinstance(preds, list):
                if any('CrSiTe3' in p for p in preds):
                    is_positive = True
            elif isinstance(preds, str):
                if 'CrSiTe3' in preds:
                    is_positive = True
            
            # 找到目标文件后即可停止
            break
    
    return is_positive

@app.post("/predict")
async def predict(filename: str = Body(...), content: str = Body(...)):
    # 0. 预测前清空 Spectra 目录，确保环境干净
    cleanup_spectra()

    # 1. 将接收到的文本内容保存为 Spectra 目录下的单个文件
    file_path = os.path.join(SPECTRA_DIR, filename)
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)
    
    # 2. 执行预测
    success = run_prediction()
    
    if not success:
        cleanup_spectra()
        return {"error": "Prediction failed"}
    
    # 3. 解析结果
    result = parse_results(filename)
    
    # 4. 清空 Spectra 目录，为下一次预测做准备
    cleanup_spectra()
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
