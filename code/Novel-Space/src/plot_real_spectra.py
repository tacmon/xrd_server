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
plot_real_spectra.py
====================
读取 ./Novel-Space/Spectra 中的所有测试数据，
复用 run_xrd_model.py 中的插值和归一化处理（对齐到 20-60度，4501个点），
并将其输出为曲线形图表，保存在 ./Novel-Space/figure/real_data 下面。

用法：
    cd /root/xrd/XRD-1.0/Novel-Space
    python plot_real_spectra.py
"""


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
# import joblib  # No longer needed as constants are hardcoded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def load_spectrum(filepath, angle_grid):
    """
    Revised loading logic consistent with run_CNN.py:
    - Supports comments (* and #)
    - Robust fallback for manual parsing
    - Encoding safe (errors='ignore')
    """
    try:
        data = np.loadtxt(filepath, comments=('*', '#'))
    except Exception:
        # Fallback to robust parsing for files with non-standard headers
        raw_data = []
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        x_val, y_val = float(parts[0]), float(parts[1])
                        raw_data.append([x_val, y_val])
                except (ValueError, IndexError):
                    continue
        data = np.array(raw_data)

    # 兼容单列（仅强度）和双列（角度 + 强度）格式
    if data.ndim == 1:
        intensity_raw = data.astype(float)
        if len(intensity_raw) == len(angle_grid):
            intensity_interp = intensity_raw
        else:
            from scipy.signal import resample
            intensity_interp = resample(intensity_raw, len(angle_grid))
    else:
        angles    = data[:, 0].astype(float)
        intensity = data[:, 1].astype(float)

        mask = (angles >= angle_grid[0] - 0.1) & (angles <= angle_grid[-1] + 0.1)
        angles    = angles[mask]
        intensity = intensity[mask]

        if len(angles) < 2:
            raise ValueError(f"文件 {filepath} 中有效角度范围内数据点不足。")

        intensity_interp = np.interp(angle_grid, angles, intensity)

    # 最大值归一化
    max_val = intensity_interp.max()
    if max_val > 0:
        intensity_interp = intensity_interp / max_val

    return intensity_interp

def main():
    
    
    spectra_dir = "Spectra"
    out_dir = os.path.join("figure", "real_data")
    
    if not os.path.exists(spectra_dir):
        print(f"[ERROR] 找不到目录: {spectra_dir}")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Hardcoded基准网格配置 (不再依赖 Model_ML.pkl)
    min_angle = 20.0
    max_angle = 60.0
    n_points = 4501
    
    angle_grid = np.linspace(min_angle, max_angle, n_points)
    
    # 支持各种后缀名 (.txt, .xy, .gk)
    valid_exts = ['.txt', '.xy', '.gk']
    spectrum_files = sorted([
        f for f in os.listdir(spectra_dir)
        if any(f.lower().endswith(ext) for ext in valid_exts) and not f.startswith('.')
    ])
    
    print(f"[INFO] 找到 {len(spectrum_files)} 个真实光谱文件，开始画图...")
    
    success_count = 0
    for fname in spectrum_files:
        fpath = os.path.join(spectra_dir, fname)
        try:
            # 执行插值 + 归一化
            spec_interp = load_spectrum(fpath, angle_grid)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(angle_grid, spec_interp, color='darkorange', linewidth=1.2)
            ax.set_xlim(min_angle, max_angle)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('2θ (degree)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_title(f'Real Measured XRD: {fname} (Interpolated)')
            
            plt.tight_layout()
            # 确保输出文件后缀为 .png
            base_name = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, base_name + '.png')
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            
            success_count += 1
            if success_count % 5 == 0:
                print(f"  ...已完成 {success_count}/{len(spectrum_files)} 张图")
                
        except Exception as e:
            print(f"[WARNING] 处理文件 {fname} 时出错: {e}")
            
    print(f"\n[DONE] 真实数据的预处理光谱图均已保存！")
    print(f"       共输出 {success_count} 张图像存放到: {os.path.abspath(out_dir)}")

if __name__ == '__main__':
    main()
