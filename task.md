# 任务说明

## 纲要
将本项目改造为“可立即打包发送给同事用来部署到服务器提供API服务”的状态。
并完成本地的API服务部署和测试。

## 任务说明素材
本章的素材旨在帮助你理解项目背景、功能需求、技术栈、部署要求等信息，请仔细阅读，并严格按照要求完成任务。

### 1. 任务支持
人物介绍：
领导A——汇总同学们的封装文件，统一部署到服务器上
领导B——前端，用户侧，需要API调用脚本和模型性能评测（封装之后其实使用起来和openai的api调用大模型的方式类似，然后把待测数据都调用一遍之后，输出一个总的性能评测报告主要就是准确率和F1值）

#### 领导A指示
docker login ccr.ccs.tencentyun.com --username=100009659750
如果镜像有更新，把镜像推到这个地址，然后再群里跟我说一声，这样就不用给我单独发tar包了
密码是：145011Clj

#### 领导A指示
你的API还需要加一个接口。我给你实现了，如下，你同步到你的代码中去（以下代码是示例，重点是predict_json方法）

```python
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
app = FastAPI()

# 容器内的工作目录
WORKING_DIR = "/workspace"
# 保持与 Dockerfile 和需求文档一致
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
            
            # 判断是否包含 CrSiTe3 (正例判断逻辑：固定检测 CrSiTe3)
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

class PredictRequest(BaseModel):
    '''
    {
        "2theta": [10.0, 20.0, 30.0, 40.0],
        "intensity": [150, 2500, 1200, 800]
    }
    '''
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
    try:
        lines = [f"{t}\t{i}" for t, i in zip(req.theta2, req.intensity)]
        content = "\n".join(lines)
        return await predict(content)
    except Exception as e:
        return PredictResponse(**{{"code": 500, "status": "error", "message": str(e), "data": None}})

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"code": 500, "status": "error", "message": str(exc), "data": None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 领导A指示
用户调用API后，得到的输出规范如下
```json
{"code": 200|500, "status": "success|error", "message":"一般消息", "data": null}
```

#### 领导A指示
群里的同学，开发服务或者做镜像的时候我给个小小的建议：
1. 新增服务时尽量在原服务基础上增加，比如原来的服务是localhost:5000/api/xxx服务，那么增加一个服务尽量保持在同端口，例如localhost:5000/api/yyy。对于需要另起端口的服务，应该是两个功能划分区别很大的场景（功能边界很大，大到可以是两个系统的情况），这样可以减少端口的资源消耗。
2. 分发镜像时，尽量包含docker-compose.yaml。如果docker-compose.yaml中使用了build，那么需要发给我完整的代码和数据，因为这是重新编译镜像。如果是通过docker save -o 导出的镜像，则docker-compose.yaml中无需加上build，因为已经有完整的镜像。但务必注意docker-compose使用了volume映射的情况，volume将本地数据映射进容器内，而且服务本身还依赖这些映射的文档或者数据，那么务必也要将这部分内容发给我。一般这种需要volume映射的情况是文档或者数据特别大，不想造成分发的镜像特别臃肿，对于小数据、小文档可以直接在编译镜像时通过Dockerfile中的COPY命令复制进镜像即可。[微笑]

#### 领导B指示
用户一条一条地调用API，将待测数据（一条）通过文本形式通过API发送，然后服务端接收到数据之后，对这一条数据进行模型预测（服务端内部如何进行文件操作，用户不关心），然后将预测结果返回给用户，目前只需要返回“是AlN 或者 不是AlN，也就是true和false”（当然，也要按照输出地json规范给用户）。

### 2. 技术路线初步设想

#### 2.1 整体架构
通过dockerfile、build、docker-compose等文件（不一定都用上，请你按照专业的知识来选择最合适的方式）来构建一个服务镜像&容器，在本宿主机中模拟出服务端；
在user_code目录下模拟用户端，写出调用服务端的代码，并完成评估脚本。

#### 2.2 用户端调用脚本
形式为一个python脚本，将API保存为.env文件，python脚本读取.env文件中的API，然后调用API，将预测结果返回给用户。
例如脚本中实现一个方法，方法的参数包含待测数据（文本格式，且不是路径而是内容）和API值。
如果我没有理解错，这个方法应当是固定的，至于用户如何读取数据、如何利用预测结果，这可以是该脚本主函数的内容了。
然后用户在主函数内读取待测数据，调用该方法，将预测结果返回给用户。
该环节就涉及到用户的数据了，用户会提供一个文件夹，里面包含所有的待测数据，事实上就是当前code/Novel-Space/soft_link/Spectra/final中的所有文件（甚至会保留子目录结构，所以需要递归读取）。
例如该代码：
```python
import os
import requests
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="XRD 模型调用脚本")
    parser.add_argument("--folder", type=str, default="./xrd识别_测试集", help="存放待测 XRD .txt 文件的目录")
    parser.add_argument("--api", type=str, default="http://localhost:8000/predict", help="API 预测地址")
    parser.add_argument("--output", type=str, default="our_model_results.json", help="结果保存路径")
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
```
其实还需要实现一个脚本，上文中的脚本，是通过API调用我自己的模型得到结果，而我现在描述的这个脚本，是通过API调用通用大模型来得到结果，而且两个脚本调用API的方式是完全一样的，只是API地址不同而已。我之前实现了一版，请直接拿来用，并且上文中所描述的脚本在“处理API返回值”等方面于其保持一致。
```python
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
        "您的任务是根据提供的 XRD 光谱数据（2-theta 角度 vs 相对强度），识别样品中是否包含 'CrSiTe3' 物相。\n\n"
        "【输入说明】\n"
        "我将提供原始的 XRD 数据采样点（2-theta vs 相对强度）。请分析这些数据以判断物相。\n\n"
        "【专业背景知识】\n"
        "CrSiTe3 属于三方晶系，空间群 R-3。其 XRD 特征峰通常出现在以下 2-theta 位置（Cu K-alpha 辐射）：\n"
        "- 约 13.0° 对应 (003) 晶面，这是极强的特征峰。\n"
        "- 约 26.1° 对应 (006) 晶面。\n"
        "- 约 39.5° 对应 (009) 晶面。\n"
        "- 其他特征峰包括 18.8° (101), 31.0° (113) 等。\n"
        "如果数据中在 13.0° 附近有非常明显的强峰，且符合层状结构的规律，则大概率包含 CrSiTe3。\n\n"
        "【输出要求】\n"
        "1. 请直接分析给定的数据内容。\n"
        "2. 最终输出必须是一个严格的 JSON 对象，不得包含任何文字说明、Markdown 标记或注释。\n"
        "3. JSON 格式：{\"result\": true} 或 {\"result\": false}\n"
        "4. 其中 true 表示识别出 CrSiTe3，false 表示未识别出。"
    )

    user_content = f"### 数据编号: {sample_id}\n数据内容:\n{data_content}\n---"

    return system_prompt, user_content

def main():
    parser = argparse.ArgumentParser(description="XRD 通用大模型逐个调用工具")
    parser.add_argument("--folder", type=str, required=True, help="包含 .txt 文件的文件夹路径")
    parser.add_argument("--api", type=str, default="https://api.siliconflow.cn/v1/chat/completions", help="API URL")
    parser.add_argument("--key", type=str, default="记得更换为自己的api", help="API Key")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="模型名称")
    parser.add_argument("--output", type=str, default="llm_output_results.json", help="输出的 JSON 文件名")
    parser.add_argument("--delay", type=float, default=1.0, help="处理每个文件后的延迟时间（秒），防止 API 频率限制")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"错误: 文件夹 {args.folder} 不存在。")
        sys.exit(1)
        
    txt_files = sorted([f for f in os.listdir(args.folder) if f.endswith(".txt")])
    if not txt_files:
        print("在指定文件夹中未找到 .txt 文件。")
        sys.exit(1)
        
    print(f"共发现 {len(txt_files)} 个文件。正在逐个处理（API: {args.api}, Model: {args.model}）...")
    
    results = {}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.key}"
    }

    for idx, filename in enumerate(txt_files):
        print(f"[{idx+1}/{len(txt_files)}] Processing {filename}...", end=" ", flush=True)
        
        file_path = os.path.join(args.folder, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_content = f.read().strip()
        except Exception as e:
            print(f"Failed to read {filename}: {e}")
            results[filename] = "Read Error"
            continue
        
        if not data_content:
            print("Skipped (Empty data).")
            results[filename] = False
            continue

        sys_prompt, user_prompt = get_high_quality_prompt(filename, data_content)
        
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }

        try:
            response = requests.post(args.api, headers=headers, json=payload, timeout=60)
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
                # 兼容返回格式可能是 {"result": true} 或直接是 true
                if isinstance(id_results, dict) and "result" in id_results:
                    results[filename] = id_results["result"]
                else:
                    results[filename] = id_results
                print("Done.")
            except json.JSONDecodeError:
                print("Error (JSON Parse Failed).")
                results[filename] = "Parse Error"
                
        except Exception as e:
            print(f"Failed ({e})")
            results[filename] = "API Error"

        # 暂停一段时间，避免触发频率限制
        if idx < len(txt_files) - 1:
            time.sleep(args.delay)

    # 保存最终结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"\n处理完成！结果已保存至 {args.output}")

if __name__ == "__main__":
    main()
```


#### 2.3 用户端评估脚本
形式为1个python脚本，用户调用API评估完待测数据得到csv后，根据csv中各个数据的真实标签和预测标签计算准确率和F1值。

我之前实现过一版，应该大差不大，或者说，尽量不要修改
``` python
import json
import argparse
import os
import sys

def get_ground_truth(filename):
    """
    根据用户最新要求：完全满分等价于“文件名中包含 AlN 的，是 true；不包含 AlN 的是 false”。
    """
    return 'AlN' in filename

def calculate_metrics(results):
    tp = 0 # True Positive
    tn = 0 # True Negative
    fp = 0 # False Positive (第一类错误: 误报)
    fn = 0 # False Negative (第二类错误: 漏报)

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
```

#### 2.4 服务端提供API服务
通过前几章你应该已经理解了用户如何使用我们的服务，现在我们就将code目录中所包含的“运行环境”、“模型文件”、“脚本”等都封装成我们的服务，用户只需要调用我们的API，就可以使用我们的服务。

**镜像构建**
1. 为了我本地快速检验，所以请使用基础镜像为我本地的pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime镜像；但是提交给领导A的时候，需要改为2.4.1-cuda12.1-cudnn9-runtime（我已经验证过，这是不影响我们的模型是否能正常运行的）
2. 构建镜像需要将code目录下的所有文件都复制到镜像中，包括运行环境、模型文件、脚本等；请注意是“复制！”，不是通过共享卷共享一下就行！
3. 在code目录下，运行pip install -e .就可以得到完整的运行环境了

**API服务提供**
1. 准备工作：进入到code/Novel-Space目录，并且
1. 用户调用API后，会传过来一个待测数据的文本