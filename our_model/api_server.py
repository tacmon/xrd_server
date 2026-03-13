import os
import shutil
import subprocess
import csv
import ast
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from typing import List
import uvicorn

app = FastAPI()

# 容器内的工作目录
WORKING_DIR = "/workspace"
SPECTRA_DIR = os.path.join(WORKING_DIR, "Spectra")
RESULT_CSV = os.path.join(WORKING_DIR, "result.csv")

# 确保 Spectra 目录存在
os.makedirs(SPECTRA_DIR, exist_ok=True)

def cleanup_spectra():
    """预测完成后清空 Spectra 目录"""
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

def parse_results():
    """解析 result.csv 并生成要求的 JSON 格式"""
    results_json = {}
    if not os.path.exists(RESULT_CSV):
        return results_json

    with open(RESULT_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['Filename']
            preds = []
            try:
                # 解析字符串形式的列表，例如 "['CrSiTe3_148', 'SiTe2_164']"
                preds = ast.literal_eval(row['Predicted phases'])
            except:
                preds = row['Predicted phases']
            
            # 判断是否包含 CrSiTe3 (正例判断逻辑参考了评测脚本)
            is_positive = False
            if isinstance(preds, list):
                if any('CrSiTe3' in p for p in preds):
                    is_positive = True
            elif isinstance(preds, str):
                if 'CrSiTe3' in preds:
                    is_positive = True
            
            results_json[filename] = is_positive
    
    return results_json

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    # 1. 保存上传的文件到 Spectra 目录
    for file in files:
        file_path = os.path.join(SPECTRA_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # 2. 执行预测
    success = run_prediction()
    
    if not success:
        cleanup_spectra()
        return {"error": "Prediction failed"}
    
    # 3. 解析结果
    results = parse_results()
    
    # 4. 清空 Spectra 目录 (作为背景任务或直接在此处执行)
    cleanup_spectra()
    
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
