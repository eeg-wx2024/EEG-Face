from imports import * # Assuming this contains necessary imports like torch, AdamW, KFold, SummaryWriter, F, np, glob etc.
from losses import SupConLoss
from collections import defaultdict
import random
# from XAI.visualization import visualize_cam # Commented out if not used, or ensure it's available
import matplotlib.pyplot as plt
import os
import logging
from mne_reader import load_and_preprocess_data, find_edf_and_markers_files
from sklearn.model_selection import train_test_split
import datetime

# Placeholder for imports.py content if needed for context, e.g.:
# import torch
# import torch.nn.functional as F
# from torch.optim import AdamW
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.model_selection import KFold
# import numpy as np
# import glob

# Placeholder for other undefined functions/classes (ensure these are defined in your environment)
# class ProNet(torch.nn.Module): def __init__(self): super().__init__(); self.fc = torch.nn.Linear(1,1) # dummy
# class OldNet(torch.nn.Module): def __init__(self, model_name): super().__init__(); self.fc = torch.nn.Linear(1,1) # dummy
# class EEGDataset(torch.utils.data.Dataset):
#     def __init__(self, data, labels): self.data = data; self.labels = labels
#     def __len__(self): return len(self.data)
#     def __getitem__(self, idx): return self.data[idx], self.labels[idx]
# def fix_random_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
# def get_args():
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpus', type=str, default='0')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--epochs', type=int, default=10) # Default to 10 for testing
#     return parser.parse_args([]) # Use [] for notebooks/testing, or sys.argv[1:] for CLI
# def get_gpu_usage(gpus_str): return 0 # dummy
# def get_log_dir(): return "runs/dummy_log_dir" # dummy for SummaryWriter
# def print_result(results): print(f"Final results: {results}") # dummy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Moved global for getconditions

def select_model(model_name):
    # classifier_EEGNet, classifier_SyncNet, classifier_EEGChannelNet, classifier_MLP, classifier_CNN
    # eeg, sync, chan, mlp, cnn
    if "pronet" in model_name:
        model = ProNet() # Placeholder
    else:
        model = OldNet(model_name=model_name) # Placeholder
    return model

def getconditions(x):
    # 算每个图片的最大值和最小值
    # 假设 images 是一个形状为 (batch_size, channels, height, width) 的张量
    max_values = x.amax(dim=[2, 3])  # 在高和宽维度上求最大值
    min_values = x.amin(dim=[2, 3])  # 在高和宽维度上求最小值
    ranges = max_values - min_values  # 计算范围
    condition = ranges < 100  # 范围小于中位数的图片
    condition = condition.to(device).squeeze()  # 转为 1D 张量
    return condition

def ratio_file_paths(file_paths, ratio=0.5):
    category_files = defaultdict(list)
    for path in file_paths:
        category = path.split('_')[-1].split('.')[0]
        category_files[category].append(path)
    
    selected_files = []
    for category, files in category_files.items():
        num_to_select = int(len(files) * ratio)
        selected_files.extend(files[:num_to_select])
    return np.array(selected_files)


def class_number_file_paths(file_paths, num_categories, random_selection=True, seed=42):
    if seed is not None:
        random.seed(seed)
    
    category_files = defaultdict(list)
    for path in file_paths:
        try:
            category = path.split('_')[-1].split('.')[0]
            category_files[category].append(path)
        except IndexError:
            raise ValueError(f"Filename '{path}' does not match the expected pattern.")
    
    total_categories = len(category_files)
    if num_categories > total_categories:
        raise ValueError(f"Requested {num_categories} categories, but only {total_categories} are available.")
    
    categories = list(category_files.keys())
    if random_selection:
        selected_categories = random.sample(categories, num_categories)
    else:
        selected_categories = categories[:num_categories]
    
    selected_files = []
    for category in selected_categories:
        selected_files.extend(category_files[category])
    return np.array(selected_files)

def get_unique_folder(base_log_dir, fold, epoch): # Modified to accept base_log_dir
    # 通过fold和epoch生成唯一的文件夹名称
    return os.path.join(base_log_dir, f"fold_{fold}_epoch_{epoch}_visualizations")

def merge_data_by_label(x, y, group_size=10):
    unique_labels = torch.unique(y)
    new_x = []
    new_y = []
    
    for label_val in unique_labels: # Renamed label to label_val to avoid conflict
        mask = (y == label_val)
        x_group = x[mask]
        # y_group = y[mask] # Not directly used
        
        num_samples = x_group.shape[0]
        num_groups = (num_samples + group_size - 1) // group_size
        
        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, num_samples)
            x_subgroup = x_group[start:end]
            x_merged = torch.mean(x_subgroup, dim=0)
            new_x.append(x_merged)
            new_y.append(label_val) # Use the current label_val
    
    if not new_x: # Handle case where no data is processed
        return torch.empty(0, *x.shape[1:], device=x.device), torch.empty(0, device=y.device)

    new_x = torch.stack(new_x)
    new_y = torch.tensor(new_y, device=y.device, dtype=y.dtype) # Ensure dtype matches
    
    return new_x, new_y

def main():
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S") # Added H M S for more uniqueness
    # Determine model_type from the script's filename
    script_name = os.path.basename(__file__)
    if '_' in script_name and '.' in script_name:
        model_type = script_name.split('_')[-1].split('.')[0]
    else:
        model_type = "unknown_model" # Fallback if filename format is unexpected

    # Base directory for all logs of this model type and run
    base_log_dir = os.path.join("logs", model_type, timestamp) # Main log directory
    os.makedirs(base_log_dir, exist_ok=True)

    # Tensorboard SummaryWriter log directory
    # Assuming get_log_dir() is for tensorboard and might be different or configurable
    # For consistency, we can make tensorboard log within our base_log_dir
    tensorboard_log_dir = os.path.join(base_log_dir, "tensorboard_runs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir) # Use specific dir for tensorboard

    # Log file for fold summaries and final results (original behavior)
    fold_summary_log_file = os.path.join(base_log_dir, "results.log")
    
    # *** MODIFICATION 1: Define the new log file path for epoch-wise logging ***
    # The file will be named eeg_log_10to1.log and placed inside the base_log_dir
    epoch_detailed_log_file = os.path.join(base_log_dir, "eeg_log_10to1.log")

    # Configure basic logging to also go to a file in base_log_dir
    # This captures print statements and logging.info messages
    master_run_log_file = os.path.join(base_log_dir, "run_output.log")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(master_run_log_file), logging.StreamHandler()])

    args = get_args()
    logging.info(f"Arguments: {args}") # Use logging
    
    # Global device is already set
    # device = torch.device(f"cuda:{get_gpu_usage(args.gpus)}") # Already global

    fix_random_seed(1234)

    # Data loading (using placeholder paths for now)
    # file_paths = np.array(glob.glob("/data1/share_data/purdue/s1/time_norm/*.pkl")) # Example path
    
    file_prefix = '' # Example
    base_path = '/data1/wuxia/dataset/FaceEEG2025/FaceEEG2025_export' # Example path
    # Dummy edf_files for testing if actual data loading is not run
    # edf_files = {"dummy_subject": {"edf": "dummy.edf", "markers": "dummy.markers"}}
    try:
        edf_files = find_edf_and_markers_files(base_path, file_prefix)
    except Exception as e:
        logging.error(f"Failed to find EDF files: {e}. Using dummy data for structure.")
        edf_files = {"dummy_subject": {"edf": "dummy.edf", "markers": "dummy.markers"}}


    logging.info(f'Found edf_files: {edf_files}')

    all_eeg_data_list = [] # Renamed to avoid confusion with torch.cat result
    all_labels_list = []   # Renamed
    invalid_files = []
    file_data_lengths = []

    for base_name, files in edf_files.items():
        edf_file_path = files['edf']
        label_file_path = files['markers']

        if not os.path.exists(label_file_path) and base_name != "dummy_subject": # Skip check for dummy
            logging.warning(f"Markers file for {edf_file_path} does not exist. Skipping.")
            continue
        
        # Dummy data loading for structure test
        if base_name == "dummy_subject":
            logging.info("Using dummy data for EEG processing.")
            eeg_data = torch.randn(100, 1, 64, 128) # (samples, channels, height, width)
            labels = torch.randint(1, 5, (100,))    # (samples,) with labels 1-4 for example
        else:
            eeg_data, labels = load_and_preprocess_data(edf_file_path, label_file_path)
        
        if eeg_data is None or labels is None:
            invalid_files.append(edf_file_path)
            logging.warning(f"Invalid data for {edf_file_path}. Skipping.")
            continue
        
        all_eeg_data_list.append(eeg_data)
        all_labels_list.append(labels)
        file_data_lengths.append(eeg_data.shape[0])

    if not all_eeg_data_list:
        logging.error("No valid EEG data found. Exiting.")
        return

    all_eeg_data = torch.cat(all_eeg_data_list).to(device)
    all_labels = torch.cat(all_labels_list).to(device)

    logging.info(f"Total EEG data shape: {all_eeg_data.shape}")
    logging.info(f"Total labels shape: {all_labels.shape}")

    merged_x, merged_y = merge_data_by_label(all_eeg_data, all_labels, group_size=10) # Default group_size
    logging.info(f"Merged data shape: {merged_x.shape}, Merged labels shape: {merged_y.shape}")

    if merged_x.shape[0] == 0:
        logging.error("No data after merging. Check merge_data_by_label or input data. Exiting.")
        return
    if merged_x.shape[0] < 5 : # KFold n_splits=5
        logging.error(f"Not enough samples ({merged_x.shape[0]}) for KFold with n_splits=5. Exiting.")
        return


    k_fold = KFold(n_splits=5, shuffle=True, random_state=1234) # Added random_state for reproducibility
    final_accs_recent_mean = [] # Stores recent_mean from end of each fold
    best_accs_fold = [] # Stores best_val_acc from each fold
    best_accs5_fold = [] # Stores best_top5_acc from each fold

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(merged_x)):
        logging.info(f"--- Starting Fold {fold + 1}/5 ---")
        train_data, val_data = merged_x[train_idx], merged_x[val_idx]
        train_labels, val_labels = merged_y[train_idx], merged_y[val_idx]

        # val_paths generation logic seems complex and depends on original non-merged indexing.
        # This part might need careful review if val_paths are crucial for something specific.
        # For now, it's kept as is, but its correctness after merging is questionable.
        val_paths = []
        # current_sample_offset = 0 # This logic needs to map merged_val_idx back to original files
        # for v_idx in val_idx: # val_idx are indices for merged_x
        #     # This mapping is non-trivial after merge_data_by_label.
        #     # Placeholder:
        #     val_paths.append("path_for_merged_sample_" + str(v_idx))


        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = select_model("eeg").to(device) # Example model type
        optimizer = AdamW(model.parameters(), lr=args.lr)
        # criterion = SupConLoss(temperature=0.07) # Contrastive loss, not used in current main loop's loss = loss_cls

        best_val_acc_fold = 0 # Best val acc for this fold
        best_epoch_fold = 0   # Epoch of best val acc for this fold
        best_top5_acc_fold = 0 # Best top-5 val acc for this fold
        best_epoch5_fold = 0  # Epoch of best top-5 val acc for this fold
        
        recent_acc_deque = [] # Using a list as a deque

        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0
            total_train_correct = 0
            total_train_samples = 0
            for step, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                total_train_samples += x.shape[0]
                
                optimizer.zero_grad()
                y_pred = model(x)
                
                # Ensure labels are 0-indexed if CrossEntropyLoss expects that
                # Original code: y = y - 1. This implies labels were 1-indexed.
                y_adjusted = y - 1 
                
                loss_cls = F.cross_entropy(y_pred, y_adjusted, label_smoothing=0.1)
                loss = loss_cls # Using only classification loss as per original
                
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * x.shape[0] # Weighted by batch size
                total_train_correct += (y_pred.argmax(dim=1) == y_adjusted).sum().item()

                if step % (len(train_loader) // 5 + 1) == 0: # Ensure at least 1 log
                    writer.add_scalar(f"Fold_{fold}/Step_Loss/Train_Cls", loss.item(), epoch * len(train_loader) + step)
            
            avg_train_loss = total_train_loss / total_train_samples
            avg_train_acc = total_train_correct / total_train_samples * 100
            writer.add_scalar(f"Fold_{fold}/Epoch_Loss/Train", avg_train_loss, epoch)
            writer.add_scalar(f"Fold_{fold}/Epoch_Acc/Train", avg_train_acc, epoch)

            # Validation
            model.eval()
            total_val_loss = 0
            total_val_correct_top1 = 0
            total_val_correct_top5 = 0
            total_val_samples = 0
            
            # Visualization folder for this epoch (if needed)
            # viz_save_dir = get_unique_folder(base_log_dir, fold, epoch) # Pass base_log_dir
            # os.makedirs(viz_save_dir, exist_ok=True)

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    total_val_samples += x_val.shape[0]
                    
                    # x_val.requires_grad_(True) # Only if explanations are generated during validation
                    y_val_adjusted = y_val - 1 # Adjust labels
                    
                    y_pred_val = model(x_val)
                    
                    loss_val_cls = F.cross_entropy(y_pred_val, y_val_adjusted)
                    total_val_loss += loss_val_cls.item() * x_val.shape[0] # Weighted by batch size

                    _, predicted_top1 = y_pred_val.max(1)
                    total_val_correct_top1 += (predicted_top1 == y_val_adjusted).sum().item()
                    
                    _, predicted_top5 = y_pred_val.topk(5, 1, True, True)
                    total_val_correct_top5 += (predicted_top5 == y_val_adjusted.unsqueeze(1)).sum().item()

            avg_val_loss = total_val_loss / total_val_samples
            avg_val_acc_top1 = total_val_correct_top1 / total_val_samples * 100
            avg_val_acc_top5 = total_val_correct_top5 / total_val_samples * 100
            
            recent_acc_deque.append(avg_val_acc_top1)
            if len(recent_acc_deque) > 4: # Keep last 4 accuracies
                recent_acc_deque.pop(0)
            current_recent_mean_acc = np.mean(recent_acc_deque) if recent_acc_deque else 0

            if avg_val_acc_top1 > best_val_acc_fold:
                best_val_acc_fold = avg_val_acc_top1
                best_epoch_fold = epoch
            
            if avg_val_acc_top5 > best_top5_acc_fold:
                best_top5_acc_fold = avg_val_acc_top5
                best_epoch5_fold = epoch

            writer.add_scalar(f"Fold_{fold}/Epoch_Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar(f"Fold_{fold}/Epoch_Acc/Val_Top1", avg_val_acc_top1, epoch)
            writer.add_scalar(f"Fold_{fold}/Epoch_Acc/Val_Top5", avg_val_acc_top5, epoch)
            writer.add_scalar(f"Fold_{fold}/Epoch_Acc/Best_Val_Top1", best_val_acc_fold, epoch)
            writer.add_scalar(f"Fold_{fold}/Epoch_Acc/Recent_Mean_Val_Top1", current_recent_mean_acc, epoch)

            log_string_epoch = (
                f"Fold {fold+1}, Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                f"Train Acc: {avg_train_acc:.2f}%, Val Acc Top1: {avg_val_acc_top1:.2f}%, Val Acc Top5: {avg_val_acc_top5:.2f}%, "
                f"Best Val Acc Top1: {best_val_acc_fold:.2f}% (Epoch {best_epoch_fold+1}), "
                f"Best Val Acc Top5: {best_top5_acc_fold:.2f}% (Epoch {best_epoch5_fold+1}), "
                f"Recent Mean Top1: {current_recent_mean_acc:.2f}%"
            )
            logging.info(log_string_epoch)

            # *** MODIFICATION 2: Log epoch results to the new detailed file ***
            with open(epoch_detailed_log_file, 'a+') as f_epoch_log:
                f_epoch_log.write(log_string_epoch + "\n")
        
        # End of epochs for the current fold
        final_accs_recent_mean.append(current_recent_mean_acc) # Store the last recent mean for this fold
        best_accs_fold.append(best_val_acc_fold)
        best_accs5_fold.append(best_top5_acc_fold)
        
        # Log summary for the current fold to results.log (uses the final state of this fold)
        fold_summary_string = (
            f"Fold {fold+1} Summary: Best Val Acc Top1: {best_val_acc_fold:.2f}% (Epoch {best_epoch_fold+1}), "
            f"Best Val Acc Top5: {best_top5_acc_fold:.2f}% (Epoch {best_epoch5_fold+1}), "
            f"Final Recent Mean Top1: {current_recent_mean_acc:.2f}%"
        )
        logging.info(fold_summary_string)
        with open(fold_summary_log_file, 'a+') as f_summary:
            f_summary.write(fold_summary_string + "\n")

    # End of all folds
    logging.info("--- Training Finished ---")
    final_summary_str_1 = f"Overall Best Top-1 Accuracies per Fold: {best_accs_fold}"
    final_summary_str_2 = f"Overall Best Top-5 Accuracies per Fold: {best_accs5_fold}"
    final_summary_str_3 = f"Overall Final Recent Mean Top-1 Accuracies per Fold: {final_accs_recent_mean}"
    
    logging.info(final_summary_str_1)
    logging.info(final_summary_str_2)
    logging.info(final_summary_str_3)

    if final_accs_recent_mean: # Ensure list is not empty
        logging.info(f"Mean of Final Recent Mean Top-1 Accuracies: {np.mean(final_accs_recent_mean):.2f}%")
    if best_accs_fold:
        logging.info(f"Mean of Best Top-1 Accuracies: {np.mean(best_accs_fold):.2f}%")


    with open(fold_summary_log_file, 'a+') as f_summary:
        f_summary.write("\n--- Overall Summary ---\n")
        f_summary.write(final_summary_str_1 + "\n")
        f_summary.write(final_summary_str_2 + "\n")
        f_summary.write(final_summary_str_3 + "\n")
        if final_accs_recent_mean:
             f_summary.write(f"Mean of Final Recent Mean Top-1 Accuracies: {np.mean(final_accs_recent_mean):.2f}%\n")
        if best_accs_fold:
            f_summary.write(f"Mean of Best Top-1 Accuracies: {np.mean(best_accs_fold):.2f}%\n")


    # print_result(final_accs_recent_mean) # Assuming print_result is a custom function for display
    writer.close()
    logging.info(f"All logs saved in: {base_log_dir}")

if __name__ == '__main__':
    # These are placeholders, ensure they are correctly defined or imported
    # For the script to run, these dummy/placeholder classes and functions need to be
    # replaced with your actual implementations from imports.py, losses.py, mne_reader.py etc.
    class ProNet(torch.nn.Module): 
        def __init__(self, input_channels=1, num_classes=40): # Example: 64 channels, 40 classes
            super().__init__()
            # Example EEGNet-like structure (very simplified)
            self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=(1, 32), padding=(0, 16))
            self.bn1 = torch.nn.BatchNorm2d(16)
            self.depthwise_conv = torch.nn.Conv2d(16, 32, kernel_size=(64, 1), groups=16, padding=(0,0)) # Assuming 64 channels
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.pointwise_conv = torch.nn.Conv2d(32, 32, kernel_size=1)
            self.bn3 = torch.nn.BatchNorm2d(32)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=(1, 8)) # Adjust pooling based on feature map size
            self.dropout = torch.nn.Dropout(0.5)
            # Calculate flattened size:
            # After conv1 (1,32) pad (0,16) on (B,1,64,128) -> (B,16,64,128)
            # After depthwise (64,1) on (B,16,64,128) -> (B,32,1,128) (Error if input is 64 channels, depthwise is on channels)
            # Let's assume input is (B, C, H, W) e.g. (B, 64, 1, T) for EEG time-series like
            # Or (B, 1, Electrodes, Timepoints) e.g. (B, 1, 64, 128)
            # If input is (B, 1, 64, 128)
            # conv1 (1,32) -> (B, 16, 64, 128)
            # depthwise_conv (input 16, output 32, kernel (64,1), groups 16) -> (B, 32, 1, 128)
            # pointwise_conv (input 32, output 32, kernel 1) -> (B, 32, 1, 128)
            # avg_pool (1,8) -> (B, 32, 1, 16)
            # Flattened: 32 * 1 * 16 = 512
            self.fc = torch.nn.Linear(32 * 1 * 16, num_classes) # Adjust 512 based on actual output of conv layers

        def forward(self, x): # x: (batch, channels, height, width) e.g. (B, 1, 64, 128)
            x = F.elu(self.bn1(self.conv1(x)))
            # x = self.dropout(x) # Optional dropout here
            x = F.elu(self.bn2(self.depthwise_conv(x)))
            x = F.elu(self.bn3(self.pointwise_conv(x)))
            x = self.avg_pool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    class OldNet(ProNet): # For simplicity, OldNet can be an alias or a different architecture
        def __init__(self, model_name, input_channels=1, num_classes=40):
            super().__init__(input_channels, num_classes)
            # Potentially different architecture for OldNet
            # For this example, it's the same as ProNet
            # print(f"Initialized OldNet (actually ProNet structure for this example) with model_name: {model_name}")


    class EEGDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels): self.data = data; self.labels = labels
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx], self.labels[idx]

    def fix_random_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    
    def get_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--gpus', type=str, default='0', help="GPU ids to use, e.g., '0' or '0,1'")
        parser.add_argument('--batch_size', type=int, default=16) # Smaller default for typical EEG data
        parser.add_argument('--lr', type=float, default=1e-4) # Common LR for AdamW
        parser.add_argument('--epochs', type=int, default=5) # Reduced for quick test
        # Add any other arguments your script uses
        # For testing in environments where sys.argv is not set (e.g. notebooks)
        # return parser.parse_args([]) # Use this for testing
        return parser.parse_args() # Use this for command-line execution

    def get_gpu_usage(gpus_str):
        # This is a placeholder. A real implementation would parse gpus_str and select a device.
        # For simplicity, we'll just use the first GPU if available, or CPU.
        if torch.cuda.is_available():
            gpu_ids = [int(g) for g in gpus_str.split(',') if g.strip().isdigit()]
            if gpu_ids:
                return gpu_ids[0] # Return the first GPU ID
        return "cpu" # Fallback or if no GPUs specified/available

    # get_log_dir is used by SummaryWriter.
    # In this modified script, SummaryWriter logs to a subdirectory of base_log_dir.
    # So, this function might not be strictly needed if SummaryWriter path is handled directly.
    # def get_log_dir(): return "runs/default_tensorboard_log_dir" 

    def print_result(results_list): 
        if results_list:
            print(f"Final Mean of Recent Mean Accuracies across folds: {np.mean(results_list):.2f}%")
        else:
            print("No results to print.")
    
    # Dummy mne_reader functions if not available
    if 'find_edf_and_markers_files' not in globals():
        def find_edf_and_markers_files(base_path, file_prefix):
            logging.warning("Using DUMMY find_edf_and_markers_files")
            # Returns a dict: {'subject_id': {'edf': 'path/to/edf', 'markers': 'path/to/markers'}}
            # Create some dummy file paths for structure
            dummy_edf_path = os.path.join(base_path, f"{file_prefix}dummy_s1.edf")
            dummy_markers_path = os.path.join(base_path, f"{file_prefix}dummy_s1.markers")
            
            # Create dummy files if they don't exist so os.path.exists works
            os.makedirs(base_path, exist_ok=True)
            if not os.path.exists(dummy_edf_path): open(dummy_edf_path, 'w').close()
            if not os.path.exists(dummy_markers_path): open(dummy_markers_path, 'w').close()

            return {"dummy_s1": {"edf": dummy_edf_path, "markers": dummy_markers_path}}

    if 'load_and_preprocess_data' not in globals():
        def load_and_preprocess_data(edf_file, markers_file):
            logging.warning(f"Using DUMMY load_and_preprocess_data for {edf_file}")
            # Returns (eeg_data_tensor, labels_tensor)
            # e.g., eeg_data: (num_epochs, num_channels, num_times) -> (N, 1, 64, 128) for conv2d
            # labels: (num_epochs,)
            num_samples = 50 + random.randint(0, 50) # Variable samples per file
            # Assuming data is pre-processed into shape (samples, 1, num_electrodes, num_timepoints)
            # And labels are 1-indexed as per y = y - 1 later
            dummy_eeg = torch.randn(num_samples, 1, 64, 128) 
            dummy_labels = torch.randint(1, 41, (num_samples,)) # 40 classes, 1-indexed
            return dummy_eeg, dummy_labels
            
    main()
