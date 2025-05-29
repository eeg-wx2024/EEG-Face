import os
import random
import numpy as np
import mne
from mne.io import read_raw_edf
import pyedflib

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

for file in files:
    edf_file_path = os.path.join(data_path, file)
    marker_file_path = os.path.join(data_path, file.replace('-edf', '').replace('.edf', '.Markers'))

    # 检查Markers文件是否存在
    if not os.path.exists(marker_file_path):
        print(f"Marker file for {file} not found. Skipping this file.")
        continue

    # 读取EDF文件
    signals, labels, sample_frequency = read_edf(edf_file_path)
    print(signals)
