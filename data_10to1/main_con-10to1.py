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

def merge_data_by_label(x, y, group_size=10):
    """
    将同标签类别的数据进行分组合并（求均值），生成新的数据集。
    :param x: 原始数据，形状为 (num_samples, ...)
    :param y: 原始标签，形状为 (num_samples,)
    :param group_size: 每组的大小，默认为10
    :return: 合并后的新数据 (new_x) 和新标签 (new_y)
    """
    # 将数据按标签分组
    unique_labels = torch.unique(y)
    new_x = []
    new_y = []
    
    for label in unique_labels:
        # 获取当前标签的所有数据
        mask = (y == label)
        x_group = x[mask]
        y_group = y[mask]
        
        # 计算需要分成多少组
        num_samples = x_group.shape[0]
        num_groups = (num_samples + group_size - 1) // group_size  # 向上取整
        
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, num_samples)
            
            # 提取当前子组的数据
            x_subgroup = x_group[start:end]
            
            # 对子组数据求均值
            x_merged = torch.mean(x_subgroup, dim=0)
            
            # 添加到新数据集
            new_x.append(x_merged)
            new_y.append(label)
    
    # 将列表转换为张量
    new_x = torch.stack(new_x)
    new_y = torch.tensor(new_y, device=y.device)
    
    return new_x, new_y

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

    # 假设 all_eeg_data 和 all_labels 是原始数据
    merged_x, merged_y = merge_data_by_label(all_eeg_data, all_labels)

    k_fold = KFold(n_splits=5, shuffle=True)
    final_accs = []
    best_accs = []
    best_accs5 = []
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(merged_x)):
        train_data = merged_x[train_idx]
        val_data = merged_x[val_idx]
        train_labels = merged_y[train_idx]
        val_labels = merged_y[val_idx]

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

        # 模型
        model = select_model("con").to(device)

        # 优化器
        optimizer = AdamW(model.parameters(), lr=args.lr)
        # 初始化 SupConLoss
        criterion = SupConLoss(temperature=0.07)  # 对比学习

        best_val_acc = 0
        best_epoch = 0
        best_epoch5 = 0
        best_top5_acc = 0
        threshold_acc = 13
        recent_acc = []
        for epoch in range(args.epochs):
            # 训练
            model.train()
            total_correct = 0
            total_sample = 0
            total_loss = 0
            for step, (x, y) in enumerate(train_loader):
                total_sample += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                y_pred = model(x)  #模型现在只返回预测结果
                # 确保 y_pred 需要梯度
                y = y - 1
                y_pred.requires_grad_()  # 启用梯度追踪
                loss_cls = F.cross_entropy(y_pred, y, label_smoothing=0.1)

                loss = loss_cls##################################原始
                if step % (len(train_loader) // 5) == 0:
                    writer.add_scalar(
                        "fold {}/step/loss_cls".format(fold),
                        loss.item(),
                        epoch * len(train_loader) + step,
                    )
                total_loss += loss.item()
                total_correct += (y_pred.argmax(dim=1) == y).sum().item()
                loss.backward()
                optimizer.step()
            train_loss = total_loss / len(train_loader)
            train_acc = total_correct / total_sample * 100
            writer.add_scalar("fold {}/train/loss".format(fold), train_loss, epoch)
            writer.add_scalar("fold {}/train/acc".format(fold), train_acc, epoch)

            # 验证
            model.eval()
            total_loss = 0
            total_correct_top1 = 0
            total_correct_top5 = 0  # 新增 Top-5 正确计数
            total_sample = 0
            # 使用自定义函数保存热力图，并传递文件路径
            save_dir = get_unique_folder(fold, epoch)  # 获取唯一的文件夹
            os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在
            for step, (x, y) in enumerate(val_loader):
                total_sample += x.shape[0]
                x = x.to(device)
                y = y.to(device)

                # Ensure input requires grad for explanation
                x.requires_grad_(True)
                y = y - 1
                y_pred = model(x)

                with torch.no_grad():
                    loss_cls = F.cross_entropy(y_pred, y)
                    total_loss += loss_cls.item()
                    # Top-1 准确率
                    _, predicted_top1 = y_pred.max(1)
                    total_correct_top1 += (predicted_top1 == y).sum().item()
                    # Top-5 准确率
                    _, predicted_top5 = y_pred.topk(5, 1, True, True)
                    total_correct_top5 += (predicted_top5 == y.unsqueeze(1)).sum().item()

            with torch.no_grad():
                val_loss = total_loss / len(val_loader)
                val_acc_top1 = total_correct_top1 / total_sample * 100
                val_acc_top5 = total_correct_top5 / total_sample * 100  # 计算 Top-5 准确率
                recent_acc.append(val_acc_top1)

                if val_acc_top1 > best_val_acc:
                    best_val_acc = val_acc_top1
                    best_epoch = epoch

                if val_acc_top5 > best_top5_acc:
                    best_top5_acc = val_acc_top5
                    best_epoch5 = epoch

                writer.add_scalar("fold {}/val/loss".format(fold), val_loss, epoch)
                writer.add_scalar("fold {}/val/acc_top1".format(fold), val_acc_top1, epoch)
                writer.add_scalar("fold {}/val/acc_top5".format(fold), val_acc_top5, epoch)  # 记录 Top-5 准确率
                writer.add_scalar(
                    "fold {}/val/best_acc".format(fold), best_val_acc, epoch
                )

                if len(recent_acc) > 4:
                    recent_mean = np.mean(recent_acc)
                    recent_acc.pop(0)
                else:
                    recent_mean = 0
                print(
                    f"fold {fold}, epoch {epoch}, train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, train_acc {train_acc:.2f}, val_acc_top1 {val_acc_top1:.2f}, val_acc_top5 {val_acc_top5:.2f}, best_val_acc {best_val_acc:.2f}, best_top5_acc {best_top5_acc:.2f}, recent_mean {recent_mean:.2f} (epoch {best_epoch}/{epoch}, epoch5 {best_epoch5}/{epoch})"
                )

                writer.add_scalar(
                    "fold {}/val/recent_mean".format(fold), recent_mean, epoch
                )

        final_accs.append(recent_mean)
        best_accs.append(best_val_acc)
        best_accs5.append(best_top5_acc)
        with open(log_file, 'a+') as f:
            f.write(f"fold {fold}, epoch {epoch}, train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, train_acc {train_acc:.2f}, val_acc_top1 {val_acc_top1:.2f}, val_acc_top5 {val_acc_top5:.2f}, best_val_acc {best_val_acc:.2f}, best_top5_acc {best_top5_acc:.2f}, recent_mean {recent_mean:.2f} (epoch {best_epoch}/{epoch}, epoch5 {best_epoch5}/{epoch})\n")
    with open(log_file, 'a+') as f:
        f.write(f"Top-1 Accuracies (5 folds): {best_accs}\n")
        f.write(f"Top-5 Accuracies (5 folds): {best_accs5}\n")
        f.write(f"Mean Accuracies (5 folds): {final_accs}\n")
    print_result(final_accs)



if __name__ == '__main__':
    main()
