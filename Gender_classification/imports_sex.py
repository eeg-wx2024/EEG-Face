from sklearn.model_selection import KFold
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

from oldnet_sex import OldNet
from pronet import ProNet


from utils import (
    BatchSampler,
    EEGDataset,
    collate_fn,
    get_args,
    get_gpu_usage,
    get_log_dir,
    print_result,
    fix_random_seed,
)
from torch.utils.tensorboard import SummaryWriter
