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
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd


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
    
    # 创建混淆矩阵保存目录
    matrix_dir = os.path.join(os.getcwd(), "eeg-matrix")
    os.makedirs(matrix_dir, exist_ok=True)
    
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

        # 模型
        model = select_model("eeg").to(device)

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
            
            # 收集所有预测和真实标签用于计算混淆矩阵
            all_preds = []
            all_targets = []
            
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
                    
                    # 收集预测和真实标签
                    all_preds.extend(predicted_top1.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

            with torch.no_grad():
                val_loss = total_loss / len(val_loader)
                val_acc_top1 = total_correct_top1 / total_sample * 100
                val_acc_top5 = total_correct_top5 / total_sample * 100  # 计算 Top-5 准确率
                recent_acc.append(val_acc_top1)

                if val_acc_top1 > best_val_acc:
                    best_val_acc = val_acc_top1
                    best_epoch = epoch
                    
                    # 在最佳epoch计算混淆矩阵和评估指标
                    all_preds = np.array(all_preds)
                    all_targets = np.array(all_targets)
                    
                    # 计算混淆矩阵
                    cm = confusion_matrix(all_targets, all_preds)
                    
                    # 保存混淆矩阵可视化
                    plt.figure(figsize=(20, 16))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.title(f'Fold {fold} - Confusion Matrix')
                    matrix_path = os.path.join(matrix_dir, f'confusion_matrix_fold_{fold}.png')
                    plt.savefig(matrix_path)
                    plt.close()
                    
                    # 计算每个类别的评估指标
                    class_metrics = {}
                    num_classes = len(np.unique(all_targets))
                    
                    # 打开日志文件，记录每个类别的指标
                    with open(os.path.join(log_dir, f"class_metrics_fold_{fold}.log"), 'w') as metrics_file:
                        metrics_file.write(f"Fold {fold} - Class Metrics\n")
                        metrics_file.write("="*80 + "\n")
                        
                        # 写入表头
                        metrics_file.write(f"{'Class':<8} | {'Accuracy (%)':<15} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1 Score (%)':<15}\n")
                        metrics_file.write("-"*80 + "\n")
                        
                        # 计算每个类的准确率、精确率、召回率和F1分数
                        for class_idx in range(num_classes):
                            # 二分类化处理每个类别
                            true_binary = (all_targets == class_idx)
                            pred_binary = (all_preds == class_idx)
                            
                            # 类别数据太少可能会导致无法计算，使用try-except处理
                            try:
                                precision = precision_score(true_binary, pred_binary, zero_division=0)
                                recall = recall_score(true_binary, pred_binary, zero_division=0)
                                f1 = f1_score(true_binary, pred_binary, zero_division=0)
                                
                                # 计算类别准确率
                                class_acc = accuracy_score(true_binary, pred_binary)
                                
                                class_metrics[class_idx] = {
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'accuracy': class_acc
                                }
                                
                                # 将比例转换为百分比进行显示
                                metrics_file.write(f"{class_idx+1:<8} | {class_acc*100:12.2f}% | {precision*100:12.2f}% | {recall*100:12.2f}% | {f1*100:12.2f}%\n")
                                
                            except Exception as e:
                                metrics_file.write(f"{class_idx+1:<8} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}\n")
                        
                        metrics_file.write("-"*80 + "\n")
                        
                        # 计算总体指标
                        overall_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                        overall_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
                        overall_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
                        
                        # 添加总体指标行
                        metrics_file.write(f"{'OVERALL':<8} | {val_acc_top1:12.2f}% | {overall_precision*100:12.2f}% | {overall_recall*100:12.2f}% | {overall_f1*100:12.2f}%\n")
                        
                        # 添加指标说明
                        metrics_file.write("\n\nNote: All metrics are reported as percentages (%).\n")
                        metrics_file.write("- Accuracy: Percentage of correct predictions for each class\n")
                        metrics_file.write("- Precision: Percentage of correct positive predictions out of all positive predictions\n")
                        metrics_file.write("- Recall: Percentage of correct positive predictions out of all actual positives\n")
                        metrics_file.write("- F1 Score: Harmonic mean of precision and recall\n")

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
