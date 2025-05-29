
"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
from collections import defaultdict

# 固定所有随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设定种子值
SEED = 2023
set_seed(SEED)

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/data0/xinyang/mapping/NICE_EEG_running/results/'
model_idx = 'test0'

parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=1, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

#=====================可以多时间自适应100,200,300=================
class PatchEmbedding(nn.Module):
    def __init__(self, nc=126):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 25), stride=(1, 1), padding=(0, 12)),   # -> (N, 16, 126, T)
            nn.AvgPool2d((1, 5), stride=(1, 5)),                         # -> (N, 16, 126, T//5)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, (nc, 1), stride=(1, 1)),                   # -> (N, 32, 1, T//5)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(32, 64, (1, 5), stride=(1, 2), padding=(0, 2)),    # -> (N, 64, 1, T1)
            nn.ELU(),
            nn.Conv2d(64, 128, (1, 5), stride=(1, 2), padding=(0, 2)),   # -> (N, 128, 1, T2)
            nn.ELU(),
            nn.Conv2d(128, 512, (1, 3), stride=(1, 1), padding=(0, 1)),  # -> (N, 512, 1, T3)
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),                                # -> (N, 512, 1, 1)
            Rearrange('b c h w -> b (c h w)'),                           # -> (N, 512)
        )

    def forward(self, x):
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
# # 模块定义
# class ResidualAdd(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x

class FlattenHead(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=126):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=512, proj_dim=512, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )



class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=512, proj_dim=512, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return super().forward(x)


class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 40
        self.batch_size = 256
        self.batch_size_test = 400
        self.batch_size_img = 500
        self.n_epochs = args.epoch
        self.time_len = 100

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0001
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/data0/xinyang/RichEEGData/PreprocessedEEGData'
        self.img_data_path = './dnn_feature/'
        self.test_center_path = './dnn_feature/'
        self.pretrain = False

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def zscore(self, data, axis=-1):
        return (data - data.mean(axis=axis, keepdims=True)) / (data.std(axis=axis, keepdims=True) + 1e-6)

    def zscore_torch_gpu(self, data, axis=-1):
        mean = data.mean(dim=axis, keepdim=True)
        std = data.std(dim=axis, keepdim=True)
        return (data - mean) / (std + 1e-6)
    def get_eeg_data(self, time_len):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(40)

        train_data = np.load('/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg/all_face_eeg.npz')
        # train_data = np.load('/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_eeg/all_face_eeg_shuffled.npz')
        # train_data = np.load('/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/raw_eeg/origin_eeg.npz')

        train_data = train_data['eeg_data'][..., :time_len]  # 截取最后一维前100个时间点
        # eeg_tensor = torch.tensor(train_data).cuda()
        # # 按每10个样本一组重塑形状，确保N是10的倍数
        # N = eeg_tensor.shape[0]
        # assert N % 10 == 0, "样本数量必须是10的倍数"
        # eeg_tensor = eeg_tensor.view(N // 10, 10, 1, 126, time_len)  # [N//10, 10, 1, 126, 100]
        #
        # # 取均值（或最大值）合并每组
        # # 合并方法1：取均值
        # # merged = eeg_tensor.mean(dim=1)  # shape: [N//10, 1, 126, 100]
        #
        # # 合并方法2：取最大值（可替代上面一行）
        # merged, _ = eeg_tensor.max(dim=1)
        # eeg_zscored = self.zscore_torch_gpu(merged, axis=-1)

        # train_data = train_data.cpu().numpy()

        return train_data, test_label

    # def get_image_data(self):
    #
    #     train_img_feature = np.load('/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_img_future/all_face_img_0.npz')
    #     train_img_feature = train_img_feature['features']
    #     test_img_feature = []
    #     # test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)
    #
    #     train_img_feature = np.squeeze(train_img_feature)
    #     # test_img_feature = np.squeeze(test_img_feature)
    #
    #     return train_img_feature, test_img_feature

    def get_image_data(self):
        train_img_feature_path = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_img_future/clip_features_faces_new.pt'
        # train_img_feature_path = '/data0/xinyang/train_arcface/processed_data/SZU_FACE_EEG_2025/all_img_future/arcface_futures_faces.npz'
        test_img_feature = []

        # 判断文件扩展名
        if train_img_feature_path.endswith('.pt'):
            data = torch.load(train_img_feature_path)
            train_img_feature = data['features']
            train_img_labels = data['labels']
            train_img_feature = train_img_feature.numpy() if isinstance(train_img_feature,
                                                                        torch.Tensor) else train_img_feature
        elif train_img_feature_path.endswith('.npz'):
            data = np.load(train_img_feature_path)
            train_img_feature = data['features']
            train_img_labels = data['labels']
            train_img_feature = train_img_feature.numpy() if isinstance(train_img_feature,
                                                                        torch.Tensor) else train_img_feature
        else:
            raise ValueError(f"不支持的文件格式: {train_img_feature_path}")
        #
        # # 去除多余维度（如果有）
        # train_img_feature = np.squeeze(train_img_feature)
        #
        # #===============img10选1或者取平均、最大================
        # N = train_img_feature.shape[0]
        # # assert train_img_feature.shape[0] == N, "图像数量应与 EEG 样本数量一致"
        #
        # # === 图像特征：每10个选一个（随机） ===
        # img_features_grouped = train_img_feature.reshape(N // 10, 10, -1)  # shape: [N//10, 10, feature_dim]
        #
        # # 方法1：随机选一个
        # indices = np.random.randint(0, 10, size=(N // 10,))
        # img_final = np.stack([img_features_grouped[i, idx] for i, idx in enumerate(indices)])
        #
        # # 方法2：可替换为平均或最大
        # # img_final = img_features_grouped.mean(axis=1)
        # # img_final = img_features_grouped.max(axis=1)

        return train_img_feature, train_img_labels

    def super_trial_augmentation(self, eeg_data, img_features, labels, group_size=50, seed=42):
        """
        实现 Super-trial 增强：将同一类别下的 group_size 个 EEG 融合后，分别加回每个原始 EEG 上。

        Args:
            eeg_data: np.array, shape [N, C, H, W] 例如 [N, 1, 126, 100]
            img_features: np.array, shape [N, feature_dim]
            labels: np.array, shape [N]
            group_size: int, number of trials per super-trial group
            seed: int, random seed

        Returns:
            augmented_eeg_array: np.array, shape [M, C, H, W]
            augmented_img_array: np.array, shape [M, feature_dim]
        """
        np.random.seed(seed)
        labels = np.array(labels)
        # 打散
        perm = np.random.permutation(len(labels))
        eeg_data = eeg_data[perm]
        img_features = img_features[perm]
        labels = labels[perm]

        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        augmented_eeg_list = []
        augmented_img_list = []

        for cls, indices in class_to_indices.items():
            indices = np.array(indices)
            if len(indices) < group_size:
                continue

            np.random.shuffle(indices)
            num_groups = len(indices) // group_size

            for i in range(num_groups):
                group_idxs = indices[i * group_size: (i + 1) * group_size]
                eeg_group = eeg_data[group_idxs]  # [group_size, C, H, W]
                img_group = img_features[group_idxs]  # [group_size, feature_dim]

                eeg_tensor = torch.tensor(eeg_group).float()  # 不用放GPU，只融合处理
                super_trial = eeg_tensor.sum(dim=0)  # [C, H, W]

                # z-score 归一化 super-trial
                mean = super_trial.mean(dim=-1, keepdim=True)
                std = super_trial.std(dim=-1, keepdim=True) + 1e-6
                super_trial = (super_trial - mean) / std  # [C, H, W]

                # 叠加到每一个原始样本上
                for j, eeg_sample in enumerate(eeg_tensor):
                    augmented = eeg_sample + super_trial  # [C, H, W]
                    augmented_eeg_list.append(augmented.numpy())
                    augmented_img_list.append(img_group[j])  # 保持图像不变

        # 返回增强后的 EEG 和图像
        augmented_eeg_array = np.stack(augmented_eeg_list)  # [M, C, H, W]
        augmented_img_array = np.stack(augmented_img_list)  # [M, feature_dim]

        return augmented_eeg_array, augmented_img_array

    def fuse_by_class_random_sampling(self, eeg_data, img_features, labels, group_size=10, mode='mean'):
        from collections import defaultdict
        import numpy as np
        import torch
        labels = np.array(labels)
        # 打散
        perm = np.random.permutation(len(labels))
        eeg_data = eeg_data[perm]
        img_features = img_features[perm]
        labels = labels[perm]

        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[label].append(idx)

        fused_eeg_list = []
        fused_img_list = []

        for cls, indices in class_to_indices.items():
            indices = np.array(indices)
            if len(indices) < group_size:
                print(f"[Skip] Class {cls} only has {len(indices)} samples, less than group_size={group_size}")
                continue

            np.random.shuffle(indices)
            num_groups = len(indices) // group_size

            print(f"[Info] Class {cls}: {len(indices)} samples, forming {num_groups} groups")

            for i in range(num_groups):
                group_idxs = indices[i * group_size: (i + 1) * group_size]
                eeg_group = eeg_data[group_idxs]
                img_group = img_features[group_idxs]

                eeg_tensor = torch.tensor(eeg_group).cuda()

                if mode == 'mean':
                    fused_eeg = eeg_tensor.mean(dim=0)
                elif mode == 'max':
                    fused_eeg, _ = eeg_tensor.max(dim=0)
                else:
                    raise ValueError(f"Unsupported fusion mode: {mode}")

                # 归一化
                mean = fused_eeg.mean(dim=-1, keepdim=True)
                std = fused_eeg.std(dim=-1, keepdim=True) + 1e-6
                fused_eeg = (fused_eeg - mean) / std

                fused_eeg_list.append(fused_eeg.cpu().numpy())

                # 图像：从当前组中随机选取一个
                rand_idx = np.random.randint(0, group_size)
                fused_img = img_group[rand_idx]
                fused_img_list.append(fused_img)

        # 添加检查和提示
        if len(fused_eeg_list) == 0:
            raise RuntimeError(
                "No valid EEG groups were formed. Possible reasons:\n"
                "- All classes have fewer than group_size samples\n"
                "- Input labels might be wrong\n"
                "- group_size is too large\n"
                "🛠 Try lowering group_size or check label distribution."
            )

        fused_eeg_array = np.stack(fused_eeg_list)  # [num_samples, 1, 126, 100]
        fused_img_array = np.stack(fused_img_list)  # [num_samples, feature_dim]
        return fused_eeg_array, fused_img_array

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):

        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        # train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_eeg, test_label = self.get_eeg_data(time_len=self.time_len)
        train_img_feature, train_labels = self.get_image_data()
        # train_eeg, train_img_feature = self.fuse_by_class_random_sampling(train_eeg, train_img_feature, train_labels, group_size=10, mode='mean')
        train_eeg, train_img_feature = self.super_trial_augmentation(train_eeg, train_img_feature, train_labels, group_size=500, seed=2023)

        # test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        # train_shuffle = np.random.permutation(len(train_eeg))
        # train_eeg = train_eeg[train_shuffle]
        # train_img_feature = train_img_feature[train_shuffle]

        # seed_value = 2025
        # np.random.seed(seed_value)
        # indices = np.arange(len(train_eeg))
        # np.random.shuffle(indices)
        # train_eeg = train_eeg[indices]
        # train_img_feature = train_img_feature[indices]

        #切分数据集为train, val, test
        test_eeg = torch.from_numpy(train_eeg[:40])
        val_eeg = torch.from_numpy(train_eeg[-200:])
        train_eeg = torch.from_numpy(train_eeg[40:-200])

        test_image = torch.from_numpy(train_img_feature[:40])
        val_image = torch.from_numpy(train_img_feature[-200:])
        train_image = torch.from_numpy(train_img_feature[40:-200])

        # val_eeg = torch.from_numpy(train_eeg[:740])
        # val_image = torch.from_numpy(train_img_feature[:740])
        #
        # train_eeg = torch.from_numpy(train_eeg[740:])
        # train_image = torch.from_numpy(train_img_feature[740:])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size,
                                                          shuffle=False)
        if isinstance(test_eeg, np.ndarray):
            test_eeg = torch.from_numpy(test_eeg)
        if not isinstance(test_image, torch.Tensor):
            test_image = torch.from_numpy(test_image)
        test_center = test_image

        if not isinstance(test_label, torch.Tensor):
            test_label = torch.from_numpy(test_label)

        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test,
                                                           shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()),
            lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):
                eeg = Variable(eeg.cuda().type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss
                loss = loss_cos

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.Enc_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        veeg_features = self.Enc_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1
                            os.makedirs('./model', exist_ok=True)
                            torch.save(self.Enc_eeg.module.state_dict(), './model/' + model_idx + 'Enc_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), './model/' + model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), './model/' + model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n' % (
                e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))

        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Enc_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model/' + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model/' + model_idx + 'Proj_img_cls.pth'), strict=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)

        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))

        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    args = parser.parse_args()

    num_sub = args.num_sub
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    for i in range(num_sub):
        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num + 1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))

