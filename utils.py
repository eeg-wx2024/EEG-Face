import subprocess
import numpy as np
import torch
from torch.utils.data import Sampler, Subset
from collections import defaultdict
from datetime import datetime
import os
import random
import pickle
import argparse
from torch.utils.data import Dataset
import torch.nn.functional as F  # 导入 PyTorch 的函数模块，提供卷积等操作


def getNow():
    now = datetime.now()
    current_year = now.year % 100
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    current_second = now.second
    return (
        str(current_year).zfill(2)
        + "/"
        + str(current_month).zfill(2)
        + "/"
        + str(current_day).zfill(2)
        + "/"
        + str(current_hour).zfill(2)
        + str(current_minute).zfill(2)
        + "-"
        + str(current_second).zfill(2)
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--N", type=int, default=1, help="N")
    parser.add_argument("--model", type=str, default="old", help="model name")# proNet;oldnet
    args = parser.parse_args()
    args.pid = get_pid()
    args.gpus = [5]
    return args


# class EEGDataset(Dataset):
#     def __init__(self, paths, ration=1):
#         self.filepaths = paths
#         self.labels = np.array(
#             [int(fp.split("_")[-1].replace(".pkl", "")) for fp in self.filepaths]
#         )

#     def __len__(self):
#         return len(self.filepaths)

#     def __getitem__(self, idx):
#         filepath = self.filepaths[idx]
#         with open(filepath, "rb") as f:
#             x = torch.tensor(pickle.load(f), dtype=torch.float)
#             y = self.labels[idx]
#             try:
#                 assert 0 <= y <= 39
#             except AssertionError:
#                 print(f"Error: {filepath}")
#         return x.unsqueeze(0), y


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx]
        return eeg, label


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_result(best_accs):
    # 计算均值和标准差
    mean_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)

    # 打印每个值，保留小数点后两位
    formatted_accs = [f"{acc:.2f}" for acc in best_accs]
    print("best accs:", ", ".join(formatted_accs))

    # 打印均值和标准差，保留小数点后两位，并以专业格式显示
    print(f"mean ± std: {mean_acc:.2f} ± {std_acc:.2f}")


def get_log_dir():
    log_dir = "./log_visual/" + getNow() + "/"
    return log_dir + get_pid() + "/"


def get_pid():
    pid = str(os.getpid())[-3:]
    return pid


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, N=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples_per_class = N
        if isinstance(dataset, Subset):
            self.labels = [dataset.dataset.labels[i] for i in dataset.indices]
        else:
            self.labels = dataset.labels
        self.num_batches = len(self.labels) // (
            self.batch_size * self.n_samples_per_class
        )
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        self.label_keys = list(self.label_to_indices.keys())

        repeats = (self.batch_size // len(self.label_keys)) + 1
        self.extended_label_keys = self.label_keys * repeats  # 复制label_keys

    def __iter__(self):

        # 用于存储所有批次的数组
        batches = np.empty(
            (self.num_batches, self.batch_size * self.n_samples_per_class), dtype=int
        )

        for i in range(self.num_batches):
            batch = np.empty((0,), dtype=int)
            classes = np.random.choice(
                self.extended_label_keys, self.batch_size, replace=False
            )

            for class_ in classes:
                indices = np.random.choice(
                    self.label_to_indices[class_],
                    self.n_samples_per_class,
                    replace=False,
                )
                batch = np.append(batch, indices)
            batches[i, :] = batch

        # 随机化批次
        np.random.shuffle(batches)

        for batch in batches:
            yield batch.tolist()  # DataLoader期望Python列表作为输出

    def __len__(self):
        return len(self.dataset) // (self.batch_size * self.n_samples_per_class)


# 实现一个转换函数，计算同一类别中N个样本的平均值
def collate_fn(batch, N=1):
    # 使用zip和星号操作符解压batch，直接转换为张量
    data_list, label_list = zip(*batch)  # 这将返回两个元组，分别包含所有数据和所有标签

    # 直接将列表转换为堆叠的张量
    data_tensor = torch.stack(data_list)
    labels_tensor = torch.tensor(label_list)

    # 假设每类样本数量N为10
    data_tensor = data_tensor.view(-1, N, *data_tensor.shape[1:])
    data_mean = data_tensor.mean(dim=1)

    labels_tensor = labels_tensor.view(-1, N)
    labels_mean = labels_tensor[:, 0]  # 选取每个批次中第一个标签作为代表标签

    return data_mean, labels_mean


def get_gpu_usage(gpus):
    """Returns a list of dictionaries containing GPU usage information."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )
    lines = output.strip().split("\n")
    # 分离已用内存和总内存，转换为 numpy 数组
    memory = np.array([line.strip().split(",") for line in lines], dtype=int)

    # 计算内存使用百分比
    memory_used_percentage = ((memory[:, 0] / memory[:, 1]) * 100).astype(int)

    # 更新不在 only_use 中的 GPU 的使用率为 100%

    if gpus is not None:
        mask = np.ones(len(memory_used_percentage), dtype=bool)
        mask[gpus] = False
        memory_used_percentage[mask] = 100

    print(memory_used_percentage)
    # 返回最小内存使用率的 GPU 索引
    return np.argmin(memory_used_percentage)
