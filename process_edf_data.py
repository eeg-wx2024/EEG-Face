import os
import numpy as np
import torch
from collections import defaultdict
from mne_reader import load_and_preprocess_data
import xml.etree.ElementTree as ET
from xml.dom import minidom
import mne
from mne import export

# 检查edfio是否安装
try:
    import edfio
except ImportError:
    raise ImportError(
        "edfio module is required for EDF export. Please install it with:\n"
        "pip install edfio\n"
        "or\n"
        "conda install -c conda-forge edfio"
    )

def process_edf_and_markers(edf_path, markers_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    eeg_data, labels = load_and_preprocess_data(edf_path, markers_path)
    if eeg_data is None or labels is None:
        print(f"Failed to load data from {edf_path} or {markers_path}.")
        return

    # 打印原始数据形状
    print(f"Original data shape: {eeg_data.shape}")

    # 按类别分组数据
    label_groups = defaultdict(list)
    for i, label in enumerate(labels):
        label_groups[label.item()].append(eeg_data[i])

    # 处理每组同类别数据
    processed_data = []
    processed_labels = []
    for label, data_list in label_groups.items():
        num_groups = (len(data_list) + 9) // 10  # 向上取整
        for i in range(num_groups):
            start = i * 10
            end = min((i + 1) * 10, len(data_list))
            group_data = data_list[start:end]
            avg_data = torch.mean(torch.stack(group_data), dim=0)
            processed_data.append(avg_data)
            processed_labels.append(label)

    # 转换为张量
    processed_data = torch.stack(processed_data)
    processed_labels = torch.tensor(processed_labels)

    # 针对形状 [3200, 1, 126, 500] 的特殊处理
    processed_data = processed_data.squeeze(1)  # [n_epochs, n_channels, n_times]
    processed_data = processed_data.permute(1, 0, 2)  # [n_channels, n_epochs, n_times]
    processed_data = processed_data.reshape(processed_data.shape[0], -1)  # [n_channels, n_samples]

    # 保存处理后的EDF数据
    base_name = os.path.basename(edf_path).replace("-edf.edf", "")
    output_edf_path = os.path.join(output_dir, f"same10_{base_name}-edf.edf")

    # 读取原始EDF文件获取通道信息
    raw_original = mne.io.read_raw_edf(edf_path, preload=True)
    ch_names = raw_original.ch_names
    sfreq = raw_original.info['sfreq']

    # 验证通道数量是否匹配
    if len(ch_names) != processed_data.shape[0]:
        raise ValueError(f"Channel number mismatch: EDF has {len(ch_names)} channels but data has {processed_data.shape[0]}")

    # 创建MNE的RawArray对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(processed_data.numpy(), info)

    # 保存为EDF文件
    export.export_raw(output_edf_path, raw, fmt='edf', overwrite=True)

    # 处理Markers文件（保持原始格式）
    try:
        # 创建新的XML树，保持原始格式
        root = ET.Element('{http://www.brainproducts.com/MarkerSet}MarkerSet')
        tree = ET.ElementTree(root)
        
        # 添加SamplingRate和SamplingInterval
        ET.SubElement(root, '{http://www.brainproducts.com/MarkerSet}SamplingRate').text = '2000'
        ET.SubElement(root, '{http://www.brainproducts.com/MarkerSet}SamplingInterval').text = '0.5'
        
        # 创建Markers节点
        markers = ET.SubElement(root, '{http://www.brainproducts.com/MarkerSet}Markers')
        
        # 只保留每组同类别数据的第一个Marker
        for label in processed_labels.unique():
            first_marker_idx = (labels == label).nonzero()[0][0]
            if first_marker_idx < len(markers):
                original_marker = markers[first_marker_idx]
                
                # 创建新Marker，保持原始命名空间
                marker = ET.SubElement(markers, '{http://www.brainproducts.com/MarkerSet}Marker')
                
                # 复制所有子元素
                for child in original_marker:
                    tag = child.tag.split('}')[-1]  # 获取不带命名空间的标签名
                    elem = ET.SubElement(marker, f'{{http://www.brainproducts.com/MarkerSet}}{tag}')
                    elem.text = child.text
                    for k, v in child.attrib.items():
                        elem.set(k, v)
            else:
                print(f"Warning: Marker index {first_marker_idx} out of bounds for label {label}.")

        # 保存新的Markers文件
        output_markers_path = os.path.join(output_dir, f"same10_{base_name}.Markers")
        
        # 添加XML声明和格式化
        xml_str = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding='utf-8')
        
        with open(output_markers_path, "wb") as f:
            f.write(xml_str)

        print(f"Successfully processed and saved to {output_edf_path} and {output_markers_path}")

    except ET.ParseError as e:
        print(f"XML parsing error in {markers_path}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {markers_path}: {e}")

def main():
    base_path = "/data1/wuxia/dataset/FaceEEG2025/FaceEEG2025_export"
    output_dir = os.path.join(base_path, "same10")

    for x in range(1, 7):
        edf_path = os.path.join(base_path, f"{x}_hushuhan-edf.edf")
        markers_path = os.path.join(base_path, f"{x}_hushuhan.Markers")

        if os.path.exists(edf_path) and os.path.exists(markers_path):
            process_edf_and_markers(edf_path, markers_path, output_dir)
        else:
            print(f"Files not found for x={x}: {edf_path} or {markers_path}")

if __name__ == "__main__":
    main()