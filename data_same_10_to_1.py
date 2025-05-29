import xml.etree.ElementTree as ET
import os
import random
import numpy as np
import mne
from mne.io import read_raw_edf
import pyedflib
import warnings
from collections import Counter, defaultdict

# 忽略RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 数据路径
data_path = "/data1/wuxia/dataset/FaceEEG2025/FaceEEG2025_export"
same10_data_path = os.path.join(data_path, "same10")
if not os.path.exists(same10_data_path):
    os.makedirs(same10_data_path)


# 获取所有文件列表
files = [f for f in os.listdir(data_path) if f.endswith('.edf')]

def read_edf(file_path):
    try:
        raw = read_raw_edf(file_path, preload=True, stim_channel=None)
        data = raw.get_data()
        labels = raw.ch_names
        sample_frequency = raw.info['sfreq']
        data = data.T.reshape(-1, 500, 126)  # 转换为 (n_epochs, 500, 126)
        return data, labels, sample_frequency
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

def write_edf(file_path, signal_data, signal_labels, sample_frequency):
    n_channels = len(signal_labels)
    signal_headers = [{'label': label, 'dimension': 'uV', 'sample_rate': sample_frequency, 
                       'physical_min': np.min(signal_data), 'physical_max': np.max(signal_data), 
                       'digital_min': -32768, 'digital_max': 32767, 'transducer': '', 'prefilter': ''} 
                      for label in signal_labels]
    f = pyedflib.EdfWriter(file_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    f.setSignalHeaders(signal_headers)
    f.writeSamples(signal_data)
    f.close()

def inspect_edf_header(file_path):
    try:
        with pyedflib.EdfReader(file_path) as f:
            print(f"=== EDF Header: {file_path} ===")
            print(f"Number of Channels: {f.signals_in_file}")
            print(f"Sample Frequency: {f.getSampleFrequency(0)} Hz")
            print("Channel Headers:")
            for i in range(f.signals_in_file):
                header = f.getSignalHeader(i)
                print(f"  Channel {i+1}:")
                print(f"    Label: {header['label']}")
                print(f"    Physical Min/Max: {header['physical_min']}/{header['physical_max']}")
                print(f"    Digital Min/Max: {header['digital_min']}/{header['digital_max']}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def check_edf_labels(file_path):
    try:
        raw = read_raw_edf(file_path, preload=True, stim_channel=None)
        labels = raw.ch_names
        print(f"Total Channels: {len(labels)}")
        print("Label Frequencies:")
        label_counts = Counter(labels)
        for label, count in label_counts.items():
            print(f"  {label}: {count} times")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def inspect_edf(file_path):
    try:
        raw = read_raw_edf(file_path, preload=True, stim_channel=None)
        print(f"=== EDF File: {file_path} ===")
        print(f"Channel Names: {raw.ch_names}")
        print(f"Sampling Frequency: {raw.info['sfreq']} Hz")
        print(f"Data Shape: {raw.get_data().shape} (n_channels, n_samples)")
        print(f"Signal Range (uV): Min={raw.get_data().min()}, Max={raw.get_data().max()}")
        print("First 5 Channel Names and Types:")
        for i, ch in enumerate(raw.ch_names[:5]):
            print(f"  {i+1}. {ch} (Type: {raw.get_channel_types()[i]})")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def parse_markers(marker_file_path):
    stim = []
    target_class = []
    try:
        tree = ET.parse(marker_file_path)
        root = tree.getroot()
        namespace = {'ns': 'http://www.brainproducts.com/MarkerSet'}
        for marker in root.findall('.//ns:Marker', namespace):
            marker_type = marker.find('ns:Type', namespace).text
            description = marker.find('ns:Description', namespace).text
            position = int(marker.find('ns:Position', namespace).text)
            if marker_type == 'Stimulus' and description:
                class_str = description.strip().split()[-1]
                if class_str.isdigit():
                    stim.append(position)
                    target_class.append(int(class_str) - 1)  # 转换为0-based
    except Exception as e:
        print(f"Error parsing {marker_file_path}: {e}")
    return stim, target_class

for file in files:
    edf_file_path = os.path.join(data_path, file)
    marker_file_path = os.path.join(data_path, file.replace('-edf', '').replace('.edf', '.Markers'))

    # 检查Markers文件是否存在
    if not os.path.exists(marker_file_path):
        print(f"Marker file for {file} not found. Skipping this file.")
        continue

    # 读取EDF文件
    data, labels, sample_frequency = read_edf(edf_file_path)
    if data is None:
        continue

    # 解析Markers文件，获取类别标签
    stim, target_class = parse_markers(marker_file_path)
    if not target_class:
        print(f"No valid markers found in {marker_file_path}. Skipping this file.")
        continue

    # 按类别分组信号
    class_to_indices = defaultdict(list)
    for idx, cls in enumerate(target_class):
        class_to_indices[cls].append(idx)

    # 初始化合并后的数据
    merged_data = []
    merged_labels = []
    merged_markers = []

    # 对每个类别进行10合1合并
    for cls, indices in class_to_indices.items():
        for i in range(0, len(indices), 10):
            chunk_indices = indices[i:i+10]
            if len(chunk_indices) == 10:
                average_signal = np.mean(data[chunk_indices], axis=0)
                merged_data.append(average_signal)
                merged_labels.append(cls)
                merged_markers.append(stim[chunk_indices[0]])

    if not merged_data:
        print(f"No signals merged for {file}. Skipping this file.")
        continue

    # 转换为numpy数组
    merged_data = np.array(merged_data)  # (320, 500, 126)
    merged_data = merged_data.transpose(0, 2, 1)  # 转换为 (320, 126, 500) 以适应 EDF 写入
    merged_labels = np.array(merged_labels)

    # 新的EDF文件路径
    new_edf_file_path = os.path.join(same10_data_path, 'same10_' + file)
    write_edf(new_edf_file_path, merged_data, labels, sample_frequency)

    # 保存处理后的Markers文件
    new_marker_file_path = os.path.join(same10_data_path, 'same10_' + os.path.basename(marker_file_path))
    with open(new_marker_file_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<MarkerSet xmlns="http://www.brainproducts.com/MarkerSet">\n')
        f.write('  <Markers>\n')
        for pos in merged_markers:
            f.write(f'    <Marker>\n')
            f.write(f'      <Type>Stimulus</Type>\n')
            f.write(f'      <Description>S {pos}</Description>\n')
            f.write(f'      <Position>{pos}</Position>\n')
            f.write(f'    </Marker>\n')
        f.write('  </Markers>\n')
        f.write('</MarkerSet>\n')

    print(f"Processed {file}: Merged {len(target_class)} samples -> {len(merged_labels)} samples.")

    # 示例调用
    inspect_edf_header(edf_file_path)
    check_edf_labels(edf_file_path)
    inspect_edf(edf_file_path)
