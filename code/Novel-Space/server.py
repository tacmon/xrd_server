"""
XRD 自动分析 API 服务

提供两个端点：
  POST /predict       — 接收文本格式的 XRD 数据，返回是否为 AlN
  POST /predict/json  — 接收 JSON 格式的 2theta + intensity 数组

启动方式：python server.py
"""

import os
import sys
import shutil
import subprocess
import csv
import ast
import uvicorn
from fastapi import FastAPI, Body
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator

app = FastAPI(
    title="XRD AutoAnalyzer API",
    description="XRD 自动分析服务：判断样品是否含有 AlN 物相",
    version="1.0.0"
)

# ── 路径配置 ──────────────────────────────────────────────
# 获取本脚本所在目录（Novel-Space/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 工作目录即 Novel-Space/
WORKING_DIR = SCRIPT_DIR
SPECTRA_DIR = os.path.join(WORKING_DIR, "Spectra")
RESULT_CSV = os.path.join(WORKING_DIR, "result.csv")
PROCESSED_RESULT_CSV = os.path.join(WORKING_DIR, "processed_result.csv")
SRC_DIR = os.path.join(WORKING_DIR, "src")

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


def cleanup_artifacts():
    """清理中间产物（result.csv, processed_result.csv, figure/, temp/）"""
    for f in [RESULT_CSV, PROCESSED_RESULT_CSV]:
        if os.path.exists(f):
            os.unlink(f)
    for d in [os.path.join(WORKING_DIR, "figure"), os.path.join(WORKING_DIR, "temp")]:
        if os.path.exists(d):
            shutil.rmtree(d)


def run_prediction():
    """调用 run_CNN.py 进行 XRD + PDF 分析"""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "run_CNN.py"), "--inc_pdf"],
            capture_output=True, text=True, check=True, cwd=WORKING_DIR
        )
        print("run_CNN.py output:", result.stdout)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = f"run_CNN.py failed (exit {e.returncode}). Stderr: {e.stderr}"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during CNN prediction: {str(e)}"
        print(error_msg)
        return False, error_msg


def run_process_results():
    """调用 process_results.py 处理结果（固定主要物质为 AlN_216）"""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, "process_results.py"),
             "--input", RESULT_CSV,
             "--output", PROCESSED_RESULT_CSV,
             "--main_substances", "AlN_216"],
            capture_output=True, text=True, check=True, cwd=WORKING_DIR
        )
        print("process_results.py output:", result.stdout)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        error_msg = f"process_results.py failed (exit {e.returncode}). Stderr: {e.stderr}"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during result processing: {str(e)}"
        print(error_msg)
        return False, error_msg


def parse_processed_results():
    """
    解析 processed_result.csv，判断是否识别出 AlN_216。
    该文件应只有两行（表头 + 数据），因为每次只预测一条数据。
    """
    if not os.path.exists(PROCESSED_RESULT_CSV):
        return None, "processed_result.csv not found"

    try:
        with open(PROCESSED_RESULT_CSV, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                predicted_phase = row.get('Predicted phases', '').strip()
                # 如果预测结果是 AlN_216，则为正例 (true)
                is_positive = (predicted_phase == "AlN_216")
                return is_positive, "OK"
        return False, "No data rows in processed_result.csv"
    except Exception as e:
        return None, f"Error parsing processed_result.csv: {str(e)}"


# ── API 端点 ──────────────────────────────────────────────

@app.post("/predict")
async def predict(content: str = Body(..., embed=True)):
    """
    接收文本格式的 XRD 数据（2theta\\tintensity，每行一个采样点），
    预测是否含有 AlN 物相。
    """
    # 0. 预测前清空环境
    cleanup_spectra()
    cleanup_artifacts()

    # 1. 将接收到的文本内容保存为 Spectra 目录下的文件
    filename = "input.txt"
    file_path = os.path.join(SPECTRA_DIR, filename)
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)

    # 2. 执行 CNN 预测 (run_CNN.py --inc_pdf)
    success, message = run_prediction()
    if not success:
        cleanup_spectra()
        cleanup_artifacts()
        return {"code": 500, "status": "error", "message": message, "data": None}

    # 3. 执行结果处理 (process_results.py)
    success, message = run_process_results()
    if not success:
        cleanup_spectra()
        cleanup_artifacts()
        return {"code": 500, "status": "error", "message": message, "data": None}

    # 4. 解析最终结果
    result, parse_msg = parse_processed_results()
    if result is None:
        cleanup_spectra()
        cleanup_artifacts()
        return {"code": 500, "status": "error", "message": parse_msg, "data": None}

    # 5. 清理，为下一次预测做准备
    cleanup_spectra()
    cleanup_artifacts()

    return {"code": 200, "status": "success", "message": "Prediction successful", "data": result}


class PredictRequest(BaseModel):
    """
    JSON 格式的预测请求：
    {
        "2theta": [10.0, 20.0, 30.0, 40.0],
        "intensity": [150, 2500, 1200, 800]
    }
    """
    theta2: List[float] = Field(..., description="2θ values", alias="2theta")
    intensity: List[float] = Field(..., description="Intensity values")

    # 根验证器：校验两个列表长度一致
    @model_validator(mode="after")
    def check_length_equal(self):
        if len(self.theta2) != len(self.intensity):
            raise ValueError(
                f"theta2 和 intensity 长度必须一致！"
                f"theta2 长度：{len(self.theta2)}，intensity 长度：{len(self.intensity)}"
            )
        return self


class PredictResponse(BaseModel):
    code: int
    status: str
    message: str
    data: Optional[bool]


@app.post("/predict/json", response_model=PredictResponse)
async def predict_json(req: PredictRequest):
    """
    接收 JSON 格式的 2theta + intensity 数组，转换为文本后调用预测。
    """
    try:
        lines = [f"{t}\t{i}" for t, i in zip(req.theta2, req.intensity)]
        content = "\n".join(lines)
        return await predict(content)
    except Exception as e:
        return PredictResponse(code=500, status="error", message=str(e), data=None)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"code": 500, "status": "error", "message": str(exc), "data": None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
