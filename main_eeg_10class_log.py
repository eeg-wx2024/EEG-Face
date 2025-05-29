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

# 只保留这10个类别
SELECTED_CLASSES = [35, 24, 32, 6, 27, 3, 8, 25, 38, 7]
# 建立类别到0-9的映射
def get_class_mapping():
    return {orig: idx for idx, orig in enumerate(SELECTED_CLASSES)}

def select_model(model_name):
    if "pronet" in model_name:
        model = ProNet()
    else:
        model = OldNet(model_name=model_name)
    return model

def get_unique_folder(fold, epoch):
    return os.path.join(get_log_dir(), f"fold_{fold}_epoch_{epoch}")

def main():
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    model_type = os.path.basename(__file__).split('_')[-2]  # 取log前的部分
    log_dir = os.path.join(model_type, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(os.getcwd(), "eeg_log_10class")

    fix_random_seed(1234)
    writer = SummaryWriter(log_dir=get_log_dir())
    args = get_args()
    print(args)
    device = torch.device(f"cuda:{get_gpu_usage(args.gpus)}")

    file_prefix = ''
    base_path = '/data1/wuxia/dataset/FaceEEG2025/FaceEEG2025_export'
    edf_files = find_edf_and_markers_files(base_path, file_prefix)
    print('edf_files:', edf_files)

    all_eeg_data = []
    all_labels = []
    invalid_files = []
    file_data_lengths = []
    class_map = get_class_mapping()

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

        # 只保留属于SELECTED_CLASSES的样本，并重映射标签
        mask = [(label+1) in SELECTED_CLASSES for label in labels.cpu().numpy()]
        mask = np.array(mask)
        if mask.sum() == 0:
            continue
        eeg_data = eeg_data[mask]
        labels = labels[mask]
        # 标签重映射（原始标签+1 -> 0-9）
        labels = torch.tensor([class_map[label.item()+1] for label in labels], dtype=torch.long)

        all_eeg_data.append(eeg_data)
        all_labels.append(labels)
        file_data_lengths.append(eeg_data.shape[0])

    if len(all_eeg_data) == 0:
        logging.info("No valid EEG data found.")
        return

    all_eeg_data = torch.cat(all_eeg_data)
    all_labels = torch.cat(all_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_eeg_data = all_eeg_data.to(device)
    all_labels = all_labels.to(device)

    k_fold = KFold(n_splits=5, shuffle=True)
    final_accs = []
    best_accs = []
    best_accs5 = []
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(all_eeg_data)):
        if fold != 3:
            continue  # 只算fold3
        train_data = all_eeg_data[train_idx]
        val_data = all_eeg_data[val_idx]
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]

        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = select_model("eeg").to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        criterion = SupConLoss(temperature=0.07)

        best_val_acc = 0
        best_epoch = 0
        best_epoch5 = 0
        best_top5_acc = 0
        recent_acc = []
        for epoch in range(args.epochs):
            model.train()
            total_correct = 0
            total_sample = 0
            total_loss = 0
            for step, (x, y) in enumerate(train_loader):
                total_sample += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                y_pred = model(x)
                y_pred.requires_grad_()
                loss_cls = F.cross_entropy(y_pred, y, label_smoothing=0.1)
                loss = loss_cls
                if step % (max(1, len(train_loader) // 5)) == 0:
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

            model.eval()
            total_loss = 0
            total_correct_top1 = 0
            total_correct_top5 = 0
            total_sample = 0
            for step, (x, y) in enumerate(val_loader):
                total_sample += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                x.requires_grad_(True)
                y_pred = model(x)
                with torch.no_grad():
                    loss_cls = F.cross_entropy(y_pred, y)
                    total_loss += loss_cls.item()
                    _, predicted_top1 = y_pred.max(1)
                    total_correct_top1 += (predicted_top1 == y).sum().item()
                    _, predicted_top5 = y_pred.topk(5, 1, True, True)
                    total_correct_top5 += (predicted_top5 == y.unsqueeze(1)).sum().item()

            with torch.no_grad():
                val_loss = total_loss / len(val_loader)
                val_acc_top1 = total_correct_top1 / total_sample * 100
                val_acc_top5 = total_correct_top5 / total_sample * 100
                recent_acc.append(val_acc_top1)

                if val_acc_top1 > best_val_acc:
                    best_val_acc = val_acc_top1
                    best_epoch = epoch

                if val_acc_top5 > best_top5_acc:
                    best_top5_acc = val_acc_top5
                    best_epoch5 = epoch

                writer.add_scalar("fold {}/val/loss".format(fold), val_loss, epoch)
                writer.add_scalar("fold {}/val/acc_top1".format(fold), val_acc_top1, epoch)
                writer.add_scalar("fold {}/val/acc_top5".format(fold), val_acc_top5, epoch)
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

                with open(log_file, 'a+') as f:
                    f.write(f"fold {fold}, epoch {epoch}, train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, train_acc {train_acc:.2f}, val_acc_top1 {val_acc_top1:.2f}, val_acc_top5 {val_acc_top5:.2f}, best_val_acc {best_val_acc:.2f}, best_top5_acc {best_top5_acc:.2f}, recent_mean {recent_mean:.2f} (epoch {best_epoch}/{epoch}, epoch5 {best_epoch5}/{epoch})\n")

                writer.add_scalar(
                    "fold {}/val/recent_mean".format(fold), recent_mean, epoch
                )

        final_accs.append(recent_mean)
        best_accs.append(best_val_acc)
        best_accs5.append(best_top5_acc)
    with open(log_file, 'a+') as f:
        f.write(f"Top-1 Accuracies (fold3): {best_accs}\n")
        f.write(f"Top-5 Accuracies (fold3): {best_accs5}\n")
        f.write(f"Mean Accuracies (fold3): {final_accs}\n")
    print_result(final_accs)


if __name__ == '__main__':
    main() 