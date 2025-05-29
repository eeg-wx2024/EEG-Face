from imports import *
from losses import SupConLoss
from collections import defaultdict
import random
from XAI.visualization import visualize_cam
import matplotlib.pyplot as plt
import os
import logging
from mne_reader import load_and_preprocess_data, find_edf_and_markers_files
from sklearn.model_selection import train_test_split
import datetime


def select_model(model_name):
    # classifier_EEGNet, classifier_SyncNet, classifier_EEGChannelNet, classifier_MLP, classifier_CNN
    # eeg, sync, chan, mlp, cnn
    if "pronet" in model_name:
        model = ProNet()
    else:
        model = OldNet(model_name=model_name)
    return model

def getconditions(x):
    # 算每个图片的最大值和最小值
    # 假设 images 是一个形状为 (batch_size, channels, height, width) 的张量
    max_values = x.amax(dim=[2, 3])  # 在高和宽维度上求最大值
    min_values = x.amin(dim=[2, 3])  # 在高和宽维度上求最小值
    ranges = max_values - min_values  # 计算范围
    condition = ranges < 100  # 范围小于中位数的图片
    condition = condition.to(device).squeeze()  # 转为 1D 张量
    # 统计false的个数
    # false_num = (condition == 0).sum().item()
    # if false_num > 0:
    #     print("false_num:", false_num)
    return condition

def ratio_file_paths(file_paths, ratio=0.5):
    # Dictionary to group file paths by their category
    category_files = defaultdict(list)
    
    # Group file paths by category based on the last number in the filename
    for path in file_paths:
        # Split to extract the category number at the end of the filename
        category = path.split('_')[-1].split('.')[0]
        category_files[category].append(path)
    
    # Filter files by ratio for each category
    selected_files = []
    for category, files in category_files.items():
        num_to_select = int(len(files) * ratio)  # Calculate the number of files to select
        selected_files.extend(files[:num_to_select])  # Select the specified number of files for each category
    
    return np.array(selected_files)


def class_number_file_paths(file_paths, num_categories, random_selection=True, seed=42):
    
    if seed is not None:
        random.seed(seed)
    
    # Dictionary to group file paths by their category
    category_files = defaultdict(list)
    
    # Group file paths by category based on the last number in the filename
    for path in file_paths:
        try:
            # Extract the category number at the end of the filename
            category = path.split('_')[-1].split('.')[0]
            category_files[category].append(path)
        except IndexError:
            raise ValueError(f"Filename '{path}' does not match the expected pattern.")
    
    total_categories = len(category_files)
    
    if num_categories > total_categories:
        raise ValueError(f"Requested {num_categories} categories, but only {total_categories} are available.")
    
    categories = list(category_files.keys())
    
    # Select categories
    if random_selection:
        selected_categories = random.sample(categories, num_categories)
    else:
        selected_categories = categories[:num_categories]
    
    # Retrieve all samples from selected categories
    selected_files = []
    for category in selected_categories:
        selected_files.extend(category_files[category])
    
    return np.array(selected_files)

# 为每次训练生成唯一文件夹
def get_unique_folder(fold, epoch):
    # 通过fold和epoch生成唯一的文件夹名称
    return os.path.join(get_log_dir(), f"fold_{fold}_epoch_{epoch}")

def main():
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    model_type = os.path.basename(__file__).split('_')[-1].split('.')[0]
    log_dir = os.path.join(model_type, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "results.log")
    
    fix_random_seed(1234)
    writer = SummaryWriter(log_dir=get_log_dir())
    args = get_args()
    print(args)
    device = torch.device(f"cuda:{get_gpu_usage(args.gpus)}")

    file_paths = np.array(glob.glob("/data1/share_data/purdue/s1/time_norm/*.pkl"))

    file_prefix = ''
    base_path = '/data1/wuxia/dataset/FaceEEG2025/FaceEEG2025_export'
    edf_files = find_edf_and_markers_files(base_path, file_prefix)
    print('edf_files:', edf_files)

    all_eeg_data = []
    all_labels = []
    invalid_files = []
    # 记录每个文件的数据样本数
    file_data_lengths = []

    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        if not os.path.exists(label_file_path):
            print(f"Markers file for {edf_file_path} does not exist. Skipping.")
            logging.info(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue

        eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path)
        
        if eeg_data is None or labels is None:
            invalid_files.append(edf_file_path)
            continue
        
        all_eeg_data.append(eeg_data)
        all_labels.append(labels)
        # 记录每个文件的数据样本数
        file_data_lengths.append(eeg_data.shape[0])  # eeg_data.shape[0] 是每个文件的样本数量

    if len(all_eeg_data) == 0:
        logging.info("No valid EEG data found.")
        return

    all_eeg_data = torch.cat(all_eeg_data)
    all_labels = torch.cat(all_labels)

    # 将数据移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_eeg_data = all_eeg_data.to(device)
    all_labels = all_labels.to(device)
    # print('all_eeg_data:', all_eeg_data)
    # print('all_labels:', all_labels)

    k_fold = KFold(n_splits=5, shuffle=True)
    final_accs = []
    best_accs = []
    best_accs5 = []
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(all_eeg_data)):
        train_data = all_eeg_data[train_idx]
        val_data = all_eeg_data[val_idx]
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]

        # 在 KFold 循环中生成 val_paths
        val_paths = []
        sample_count = 0  # 用来累计样本数量
        for idx in val_idx:
            # 根据样本的索引，确定它属于哪个文件
            sample_idx = idx
            for i, data_len in enumerate(file_data_lengths):  # file_data_lengths 是每个文件的数据样本数
                sample_count += data_len
                if sample_idx < sample_count:
                    base_name = list(edf_files.keys())[i]
                    val_paths.append(edf_files[base_name]['edf'])  # 获取 'edf' 路径
                    break


        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 确保数据是 PyTorch 张量并在设备上
        train_data = train_data.float().to(device)  # 直接转换（假设已是张量）
        train_labels = train_labels.long().to(device)

        # 初始化模型
        model = select_model("knn").to(device)

        # 拟合数据
        model.net.fit(train_data, train_labels)

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred_top1 = model(val_data, topk=1)
            val_pred_top5 = model(val_data, topk=5)
            
            # 计算准确率
            top1_correct = (val_pred_top1.squeeze() == val_labels).float().sum().item()
            top1_acc = top1_correct / len(val_labels) * 100
            
            top5_correct = sum(1 for i in range(len(val_labels)) 
                              if val_labels[i] in val_pred_top5[i])
            top5_acc = top5_correct / len(val_labels) * 100

        print(f"Fold {fold} - Top1: {top1_acc:.2f}%, Top5: {top5_acc:.2f}%")

        # 记录结果
        final_accs.append(top1_acc)
        best_accs.append(top1_acc)
        best_accs5.append(top5_acc)
        with open(log_file, 'a+') as f:
            f.write(f"fold {fold}, top1_acc {top1_acc:.2f}, top5_acc {top5_acc:.2f}\n")
    with open(log_file, 'a+') as f:
        f.write(f"Top-1 Accuracies (5 folds): {best_accs}\n")
        f.write(f"Top-5 Accuracies (5 folds): {best_accs5}\n")
        f.write(f"Mean Top-1: {np.mean(best_accs):.2f}\n")
        f.write(f"Mean Top-5: {np.mean(best_accs5):.2f}\n")
    print_result(final_accs)



if __name__ == '__main__':
    main()
