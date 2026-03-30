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

"""
generate_theoretical_spectra.py
===============================
读取 ./Novel-Space-CST/References 下的所有 CIF 文件，
使用 pymatgen.analysis.diffraction.xrd 计算其理论精确的 XRD 数据。

针对计算结果（一组离散的 (2θ, 强度) 峰值数据）：
考虑到真实实验谱是带有展宽的连续曲线，直接保存完全离散的理论火柴棍峰不利于 ML 模型
（因为模型学到的是带展宽的增广谱），因此我们会对其进行一次**简单的物理展宽 (Gaussian/Lorentzian)**，
生成 20°~60° 等间距 4501 个点的标准曲线数据，然后输出并覆盖到 ./Novel-Space-CST/Spectra/ 目录下
作为实验“对照组”。

用法：
    cd /root/xrd/wrh-ML/Novel-Space-CST
    python generate_theoretical_spectra.py
"""

import numpy as np

import pymatgen.core as mg
from pymatgen.analysis.diffraction.xrd import XRDCalculator



def apply_broadening(x_grid, peaks_x, peaks_y, fwhm=0.1):
    """
    对离散衍射峰应用高斯展宽，生成平滑连贯的 XRD 曲线。
    fwhm: 半高宽 (Full Width at Half Maximum)
    """
    y_profile = np.zeros_like(x_grid)
    sigma = fwhm / 2.35482  # FWHM 到 sigma 的转换公式
    
    for px, py in zip(peaks_x, peaks_y):
        # 叠加高斯峰
        y_profile += py * np.exp(-((x_grid - px) ** 2) / (2 * sigma ** 2))
        
    return y_profile

def main():
    
    
    ref_dir = "References"
    spectra_dir = "Spectra"
    
    if not os.path.exists(ref_dir):
        print(f"[ERROR] 找不到目录: {ref_dir}")
        return
        
    os.makedirs(spectra_dir, exist_ok=True)
    
    # 设定与模型一致的角度和采样数
    min_angle = 20.0
    max_angle = 60.0
    n_points = 4501
    
    angle_grid = np.linspace(min_angle, max_angle, n_points)
    xrd_calc = XRDCalculator(wavelength='CuKa')
    
    cif_files = sorted([f for f in os.listdir(ref_dir) if f.endswith('.cif')])
    print(f"[INFO] 找到 {len(cif_files)} 个参考相 CIF 文件，开始计算...")
    
    count = 0
    for cif in cif_files:
        cls_name = os.path.splitext(cif)[0]
        cif_path = os.path.join(ref_dir, cif)
        
        try:
            print(f"  -> 处理: {cls_name}")
            struc = mg.Structure.from_file(cif_path)
            
            # 获取 20-60度 之内的理论衍射峰
            pattern = xrd_calc.get_pattern(struc, two_theta_range=(min_angle, max_angle))
            
            # 获取离散的理论峰位置和强度
            peaks_theta = pattern.x
            peaks_intensity = pattern.y
            
            # 应用简单的物理高斯展宽（FWHM 设为 0.1 度比较符合仪器常理）
            Theoretical_Profile = apply_broadening(angle_grid, peaks_theta, peaks_intensity, fwhm=0.1)
            
            # 最大值归一化 (可选，这里归一化为 1.0)
            if Theoretical_Profile.max() > 0:
                Theoretical_Profile /= Theoretical_Profile.max()
            else:
                print(f"  [WARNING] 相 {cls_name} 在 {min_angle}-{max_angle}° 无任何衍射峰！")
                
            # 保存到 Spectra 目录，后缀加个 _Theoretical 做明显标识
            out_filename = f"{cls_name}_Theoretical.txt"
            out_path = os.path.join(spectra_dir, out_filename)
            
            # 存为两列格式: 2Theta Intensity
            save_data = np.column_stack((angle_grid, Theoretical_Profile))
            np.savetxt(out_path, save_data, fmt="%.5f", delimiter=" ")
            
            count += 1
            
        except Exception as e:
            print(f"  [ERROR] 处理 {cif} 失败: {e}")
            
    print(f"\n[DONE] 成功计算并处理了 {count} 个理论精确 XRD 谱！")
    print(f"       已作为对照组数据存放进入待测目录: {os.path.abspath(spectra_dir)}/")

if __name__ == '__main__':
    main()
