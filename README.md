# 材料大模型 XRD 封装组件使用说明

本项目旨在通过 Docker 容器化技术，为 XRD (X-Ray Diffraction) 材料相识别模型提供一个标准化的 API 环境模拟。用户可以通过 API 发送 XRD 谱图数据，并获取模型对特定相（本例为 `CrSiTe3`）的分类预测结果。

## 核心流程

您理解的执行顺序非常准确：
1. **构建镜像**: `cd ./docker && bash build_image.sh`
2. **启动容器**: `bash run_container.sh`
3. **模拟调用**: `cd ../our_model && python3 client.py`

---

## 脚本逻辑详细说明

### 1. 环境构建层 (`./docker/`)

*   **`Dockerfile`**: 
    *   基础镜像使用带有 CUDA 支持的 PyTorch 环境。
    *   自动安装项目所需的依赖包（`autoXRD` 库及其第三方依赖）。
    *   将模型参数 (`Models/`)、参考数据 (`References/`) 和核心逻辑脚本 (`run_CNN.py`, `api_server.py`) 封装进镜像。
    *   暴露端口 `8000` 并设置自动运行 API 服务。
*   **`build_image.sh`**: 
    *   自动化构建脚本。它会确保 `Dockerfile` 的内容正确，然后调用 `docker build` 命令生成名为 `xrd-test:latest` 的镜像。
*   **`run_container.sh`**:
    *   **关键逻辑**: 使用 `--gpus all` 参数启动容器，确保模型能调用物理 GPU 进行加速。
    *   端口映射: 将容器内的 `8000` 端口映射到宿主机的 `8000` 端口。

### 2. 服务实现层 (`./our_model/`)

*   **`api_server.py`**:
    *   **后端框架**: 基于 FastAPI 实现。
    *   **接口逻辑**: 接收用户通过 `POST /predict` 发送的文本原始数据，将其保存为临时文件，随后调用 `run_CNN.py` 进行逻辑处理。
    *   **分类判定**: 解析模型输出的 `result.csv`，专门检查预测结果中是否包含 `CrSiTe3`，并返回标准 JSON 响应。
    *   **自清理**: 每次预测结束后会自动清空 `Spectra/` 目录，确保下一次调用的环境隔离。
*   **`run_CNN.py`**:
    *   项目的核心算法入口。负责加载 CNN 模型，对指定目录下的 XRD 谱图进行相位识别。
    *   输出结果会保存到 `result.csv` 中。

### 3. 用户调用层 (`./our_model/`)

*   **`client.py`**:
    *   **模拟用户行为**: 遍历本地指定目录（默认 `./Spectra`）下的所有 `.txt` 文件。
    *   **文本交互**: 按照需求，它将文件内容读取为字符串，逐个通过 API 发送。
    *   **结果汇总**: 收集所有 API 的反馈，并最终生成一个 `results.json` 文件供用户查看。

---

## API 响应规范

所有接口调用将返回以下结构的 JSON：

```json
{
    "code": 200,      // 200 为成功，500 为错误
    "status": "success", 
    "message": "...", 
    "data": true      // true 表示检测到正例 (CrSiTe3)，false 表示未检测到
}
```

## 注意事项
- 请确保宿主机已安装 NVIDIA Container Toolkit 以支持 Docker 的 GPU 调用。
- 调用 `client.py` 前，请确保 `api_url` 指向正确的容器 IP 和端口。
