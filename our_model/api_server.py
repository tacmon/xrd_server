import os
import sys
import shutil
import subprocess
import csv
import ast
import uvicorn
from fastapi import FastAPI, Body

app = FastAPI()

# 容器内的工作目录
WORKING_DIR = "/workspace"
# 使用容器内部的临时目录，避免清空挂载的宿主机目录
SPECTRA_DIR = "/tmp/api_spectra"
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
        # 使用 sys.executable 以确保使用与当前服务器相同的 Python 解释器
        # capture_output=True 用于捕获错误日志以供诊断
        result = subprocess.run([sys.executable, "run_CNN.py", "--inc_pdf", f"--spectra_dir={SPECTRA_DIR}"], 
                                capture_output=True, text=True, check=True, cwd=WORKING_DIR)
        print("Prediction output:", result.stdout)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = f"Prediction failed with exit code {e.returncode}. Stderr: {e.stderr}"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {str(e)}"
        print(error_msg)
        return False, error_msg

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
async def predict(content: str = Body(..., embed=True)):
    # 0. 预测前清空 Spectra 目录，确保环境干净
    cleanup_spectra()

    # 1. 将接收到的文本内容保存为 Spectra 目录下的固定文件名
    filename = "input.txt"
    file_path = os.path.join(SPECTRA_DIR, filename)
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)
    
    # 2. 执行预测
    success, message = run_prediction()
    
    if not success:
        cleanup_spectra()
        return {"code": 500, "status": "error", "message": message, "data": None}
    
    # 3. 解析结果
    result = parse_results(filename)
    
    # 4. 清空 Spectra 目录，为下一次预测做准备
    cleanup_spectra()
    
    return {"code": 200, "status": "success", "message": "Prediction successful", "data": result}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"code": 500, "status": "error", "message": str(exc), "data": None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
