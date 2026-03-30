import os
import sys

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

from mp_api.client import MPRester
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

# 将 YOUR_API_KEY 替换为您在官网获取的真实秘钥
# 从 .env 文件或环境变量读取
api_key = os.getenv("MP_API_KEY")
material_id = "mp-" + input("输入ID：") # 以锐钛矿 TiO2 为例

# 连接到 MP 数据库
with MPRester(api_key) as mpr:
    # 获取材料结构 (默认获取的是 conventional standard structure)
    structure = mpr.get_structure_by_material_id(material_id)
    
    # 将结构对象导出为 CIF 文件
    output_dir = "All_CIFs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"{material_id}.cif")
    structure.to(fmt="cif", filename=filename)
    
    print(f"成功下载 {material_id} 的 CIF 文件并保存为 {filename}")
