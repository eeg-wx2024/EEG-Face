import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
from sklearn.model_selection import KFold
import logging
from datetime import datetime
import argparse
import xml.etree.ElementTree as ET

# from eeg_net import EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet

def normalize_samples(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x_normalized = (x - mean) / (std + 1e-6)
    return x_normalized

class MNEReader(object):
    def __init__(self, filetype='edf', method='stim', resample=None, length=500, exclude=(), stim_channel='auto', montage=None):
        self.filetype = filetype
        self.file_path = None
        self.resample = resample
        self.length = length
        self.exclude = exclude
        self.stim_channel = stim_channel
        self.montage = montage
        if stim_channel == 'auto':
            assert method == 'manual'

        if method == 'auto':
            self.method = self.read_auto
        elif method == 'stim':
            self.method = self.read_by_stim
        elif method == 'manual':
            self.method = self.read_by_manual
        self.set = None
        self.pos = None

    def get_set(self, file_path, stim_list=None):
        self.file_path = file_path
        self.set = self.method(stim_list)
        return self.set

    def get_pos(self):
        assert self.set is not None
        return self.pos

    def get_item(self, file_path, sample_idx, stim_list=None):
        if self.file_path == file_path:
            return self.set[sample_idx]
        else:
            self.file_path = file_path
            self.set = self.method(stim_list)
            return self.set[sample_idx]

    def read_raw(self):
        if self.filetype == 'bdf':
            raw = mne.io.read_raw_bdf(self.file_path, preload=True, exclude=self.exclude, stim_channel=self.stim_channel)
            print(raw.info['sfreq'])
        elif self.filetype == 'edf':
            raw = mne.io.read_raw_edf(self.file_path, preload=True, exclude=self.exclude, stim_channel=self.stim_channel)
        else:
            raise Exception('Unsupported file type!')
        return raw

    def read_by_manual(self, stim_list):
        raw = self.read_raw()
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
        set = []
        for i in stim_list:
            end = i + self.length
            data, times = raw[picks, i:end]
            set.append(data.T)
        return set

    def read_auto(self, *args):
        raw = self.read_raw()
        events = mne.find_events(raw, stim_channel=self.stim_channel, initial_event=True, output='step')
        event_dict = {'stim': 65281, 'end': 0}
        epochs = mne.Epochs(raw, events, event_id=event_dict, preload=True).drop_channels('Status')
        epochs.equalize_event_counts(['stim'])
        stim_epochs = epochs['stim']
        del raw, epochs, events
        return stim_epochs.get_data().transpose(0, 2, 1)

# def ziyan_read(file_path):
#     print('biaoqianpath:', file_path)
#     with open(file_path) as f:
#         stim = []
#         target_class = []
#         for line in f.readlines():
#             print(line)
#             if line.strip().startswith('Stimulus'):
#                 line = line.strip().split(',')
#                 classes = int(line[1][-2:])
#                 time = int(line[2].strip())
#                 stim.append(time)
#                 target_class.append(classes)
#     return stim, target_class

def ziyan_read(file_path):
    stim = []
    target_class = []

    try:
        # 解析 XML 文件
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 提取命名空间
        ns = {'ns': 'http://www.brainproducts.com/MarkerSet'}

        # 遍历所有 <Marker> 元素
        for marker in root.findall('.//ns:Marker', ns):  # 使用命名空间查找 Marker
            # 获取 <Type>， <Description> 和 <Position>
            marker_type = marker.find('ns:Type', ns).text
            description = marker.find('ns:Description', ns).text
            position = int(marker.find('ns:Position', ns).text)

            # 只处理 Stimulus 类型的 Marker
            if marker_type == 'Stimulus' and description:
                # 提取类别信息，通常在 Description 中以 "S" 开头
                class_str = description.strip().split()[-1]  # 获取类别，去掉前缀 "S"
                if class_str.isdigit():
                    classes = int(class_str)
                    stim.append(position)
                    target_class.append(classes)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    # 打印前几个刺激信息用于调试
    print(f"Stimulus list: {stim[:10]}")
    print(f"Target class list: {target_class[:10]}")

    return stim, target_class

def find_edf_and_markers_files(base_path, file_prefix=None):
    edf_files = {}
    for filename in os.listdir(base_path):
        if filename.endswith('.edf') and (file_prefix is None or filename.startswith(file_prefix)):
            base_name = filename[:-8]
            edf_files[base_name] = {
                'edf': os.path.join(base_path, filename),
                'markers': os.path.join(base_path, base_name + '.Markers')
            }
    return edf_files

def load_and_preprocess_data(edf_file_path, label_file_path):
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    print(f"Raw data info: {raw.info}")

    edf_reader = MNEReader(filetype='edf', method='manual', length=500)
    stim, target_class = ziyan_read(label_file_path)

    # 将标签值减1，以使标签范围从0到49
    target_class = [cls - 1 for cls in target_class]

    stim, target_class = ziyan_read(label_file_path)
    # print(f"Stimulus list: {stim}")
    # print(f"Target class list: {target_class}")
    xx = edf_reader.get_set(file_path=edf_file_path, stim_list=stim)
    print('edf_file_path:', edf_file_path)
    # print('xx:', xx)

    xx_np = np.array(xx)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_np.shape= {xx_np.shape}")

    print('____________________')
    print(f'xx_np shape is {xx_np.shape}')
    # 如果通道数不是127，跳过
    if xx_np.shape[2] != 126:
        logging.info(f"Skipping file {edf_file_path}, expected 127 channels but got {xx_np.shape[2]}.")
        return None, None

    xx_normalized = normalize_samples(xx_np)
    logging.info(f"{os.path.basename(edf_file_path)} - xx_normalized.shape= {xx_normalized.shape}")

    eeg_data = np.transpose(xx_normalized, (0, 2, 1))
    eeg_data = eeg_data[:, np.newaxis, :, :]
    logging.info(f"{os.path.basename(edf_file_path)} - eeg_data.shape= {eeg_data.shape}")

    eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    labels_tensor = torch.tensor(target_class, dtype=torch.long)
    
    return eeg_data_tensor, labels_tensor

def setup_logging(model_name):
    log_dir = os.path.join(model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'Starting training with model {model_name}')

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, help='Model to use: EEGNet, classifier_EEGNet, classifier_SyncNet, classifier_CNN, classifier_EEGChannelNet',default='classifier_EEGNet')
#     parser.add_argument('--prefix', type=str, default=None, help='File prefix to filter EEG data files')
#     args = parser.parse_args()

#     model_name = args.model
#     file_prefix = args.prefix

#     # Setup logging
#     setup_logging(model_name)

#     base_path = '/data1/wuxia/dataset/FaceEEG_new/same5'
#     edf_files = find_edf_and_markers_files(base_path, file_prefix)

#     all_eeg_data = []
#     all_labels = []
#     invalid_files = []

#     for base_name, files in edf_files.items():
#         edf_file_path = files['edf']
#         label_file_path = files['markers']

#         if not os.path.exists(label_file_path):
#             logging.info(f"Markers file for {edf_file_path} does not exist. Skipping.")
#             continue

#         eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path)
        
#         if eeg_data is None or labels is None:
#             invalid_files.append(edf_file_path)
#             continue
        
#         all_eeg_data.append(eeg_data)
#         all_labels.append(labels)

#     if len(all_eeg_data) == 0:
#         logging.info("No valid EEG data found.")
#         return

#     all_eeg_data = torch.cat(all_eeg_data)
#     all_labels = torch.cat(all_labels)

#     # 将数据移到 GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     all_eeg_data = all_eeg_data.to(device)
#     all_labels = all_labels.to(device)

#     kfold = KFold(n_splits=5, shuffle=True, random_state=42)
#     num_epochs = 300

    

#     # model = nn.DataParallel(model)  # 支持多GPU训练
#     # model = model.to(device)
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     for fold, (train_idx, test_idx) in enumerate(kfold.split(all_eeg_data)):
#         logging.info(f"FOLD {fold+1}")

#         # 实例化模型
#         if model_name == 'EEGNet':
#             model = EEGNet(n_timesteps=500, n_electrodes=127, n_classes=50)
#         elif model_name == 'classifier_EEGNet':
#             model = classifier_EEGNet(temporal=500)
#         elif model_name == 'classifier_SyncNet':
#             model = classifier_SyncNet(temporal=500)
#         elif model_name == 'classifier_CNN':
#             model = classifier_CNN(num_points=500, n_classes=50)
#         elif model_name == 'classifier_EEGChannelNet':
#             model = classifier_EEGChannelNet(temporal=500)
#         else:
#             raise ValueError(f"Unknown model: {model_name}")
        
#         # 下面两行需要放在模型初始化之后
#         model = model.to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.0001)

#         train_dataset = TensorDataset(all_eeg_data[train_idx], all_labels[train_idx])
#         test_dataset = TensorDataset(all_eeg_data[test_idx], all_labels[test_idx])

#         train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#         best_acc = 0.0
#         best_epoch = 0
#         best_acc_list = []
#         for epoch in range(num_epochs):
#             model.train()
#             running_loss = 0.0
#             correct = 0
#             total = 0
#             for inputs, labels in train_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#             epoch_loss = running_loss / len(train_loader)
#             epoch_acc = 100 * correct / total

#             model.eval()
#             correct_test = 0
#             total_test = 0
#             with torch.no_grad():
#                 for inputs, labels in test_loader:
#                     inputs, labels = inputs.to(device), labels.to(device)
#                     outputs = model(inputs)
#                     _, predicted = torch.max(outputs, 1)
#                     total_test += labels.size(0)
#                     correct_test += (predicted == labels).sum().item()

#             test_acc = 100 * correct_test / total_test

#             if test_acc > best_acc:
#                 best_acc = test_acc
#                 best_epoch = epoch
#         logging.info(
#     f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, "
#     f"best_acc: {best_acc:.2f}%, best_epoch: {best_epoch+1}"
# )

#         best_acc_list.append(best_acc)
            
#             # logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')

#     if invalid_files:
#         logging.info("Files skipped due to invalid channel size:")
#         for invalid_file in invalid_files:
#             logging.info(invalid_file)

# if __name__ == '__main__':
#     main()
