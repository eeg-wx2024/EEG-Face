import os
import shutil
import mne
import numpy as np

def safe_normalize_edf_channels(raw):
    """
    安全版本的通道归一化，确保输出值在EDF格式允许范围内
    """
    data = raw.get_data()
    
    # 按通道归一化（更严格的数值限制）
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True)
    
    # 归一化并限制在±1e3范围内（确保EDF头文件字段不溢出）
    normalized_data = (data - means) / (stds + 1e-6)
    normalized_data = np.clip(normalized_data, -1000, 1000)
    
    # 创建新的Raw对象
    return mne.io.RawArray(normalized_data, raw.info)

def process_edf_files(input_dir, output_dir):
    """
    更安全的EDF处理流程
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, 7):
        edf_file = f"{i}_hushuhan-edf.edf"
        markers_file = f"{i}_hushuhan-edf.Markers"
        input_edf_path = os.path.join(input_dir, edf_file)
        input_markers_path = os.path.join(input_dir, markers_file)
        
        if not os.path.exists(input_edf_path):
            print(f"EDF文件不存在: {input_edf_path}")
            continue

        output_edf_path = os.path.join(output_dir, edf_file)
        output_markers_path = os.path.join(output_dir, markers_file)

        print(f"\n处理文件: {edf_file}")
        
        try:
            # 1. 读取EDF（忽略时间范围外的注释）
            raw = mne.io.read_raw_edf(input_edf_path, preload=True, verbose='error')
            
            # 2. 安全归一化
            normalized_raw = safe_normalize_edf_channels(raw)
            
            # 3. 导出EDF（设置明确的物理量范围）
            normalized_raw.export(
                output_edf_path,
                fmt='edf',
                physical_range=(-1000, 1000),  # 显式设置允许范围
                overwrite=True,
                verbose='error'
            )
            
            # 4. 复制Markers文件
            if os.path.exists(input_markers_path):
                shutil.copy2(input_markers_path, output_markers_path)
                print("Markers文件复制成功")
            
            # 验证结果
            print(f"处理成功. 数据范围: [{normalized_raw.get_data().min():.2f}, {normalized_raw.get_data().max():.2f}]")
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            continue

if __name__ == "__main__":
    input_dir = "/data1/wuxia/dataset/FaceEEG2025/FaceEEG2025_export"
    output_dir = "/data1/wuxia/dataset/FaceEEG2025/normalized_data"
    process_edf_files(input_dir, output_dir)