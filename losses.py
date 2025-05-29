import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，并简写为 nn


class SupConLoss(nn.Module):  # 定义一个名为 SupConLoss 的类，继承自 nn.Module
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self, temperature=0.07, contrast_mode="all", base_temperature=0.07
    ):  # 初始化方法，定义超参数
        super(SupConLoss, self).__init__()  # 调用父类的初始化方法
        self.temperature = temperature  # 初始化温度参数
        self.contrast_mode = contrast_mode  # 初始化对比模式（默认是 'all'）
        self.base_temperature = base_temperature  # 初始化基础温度参数

    def forward(self, features, labels=None, mask=None):  # 前向传播方法，用于计算损失
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device  # 获取设备信息

        if len(features.shape) < 3:  # 检查输入特征的维度，如果小于 3 则抛出错误
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required"
            )
        if len(features.shape) > 3:  # 如果特征维度大于 3，则调整形状
            features = features.view(
                features.shape[0], features.shape[1], -1
            )  # 将特征调整为 [batch_size, n_views, -1]

        batch_size = features.shape[0]  # 获取批次大小
        if (
            labels is not None and mask is not None
        ):  # 如果同时传入了标签和掩码，则抛出错误
            raise ValueError("Cannot define both `labels` and `mask`")
        elif (
            labels is None and mask is None
        ):  # 如果标签和掩码都未传入，则使用对角矩阵作为掩码
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  # 如果传入了标签，则生成掩码
            labels = labels.contiguous().view(-1, 1)  # 将标签调整为列向量
            if labels.shape[0] != batch_size:  # 如果标签数量与特征数量不匹配，抛出错误
                raise ValueError("Num of labels does not match num of features")
            mask = (
                torch.eq(labels, labels.T).float().to(device)
            )  # 生成掩码：如果标签相同，掩码值为1
        else:  # 如果传入了掩码，则直接使用
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # 获取对比样本的数量
        contrast_feature = torch.cat(
            torch.unbind(features, dim=1), dim=0
        )  # 将特征在第二个维度展开并拼接

        if self.contrast_mode == "one":  # 如果对比模式是 'one'
            anchor_feature = features[:, 0]  # 只取第一个视角作为锚点特征
            anchor_count = 1  # 锚点数量为1
        elif self.contrast_mode == "all":  # 如果对比模式是 'all'
            anchor_feature = contrast_feature  # 所有特征都作为锚点
            anchor_count = contrast_count  # 锚点数量为对比数量
        else:  # 如果对比模式未知，则抛出错误
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(  # 计算锚点特征与对比特征的点积，并除以温度参数
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(
            anchor_dot_contrast, dim=1, keepdim=True
        )  # 为了数值稳定性，减去每行的最大值
        logits = anchor_dot_contrast - logits_max.detach()  # 更新 logits

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # 扩展掩码
        # mask-out self-contrast cases
        logits_mask = torch.scatter(  # 创建 logits 掩码，防止自我对比
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask  # 更新掩码

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 计算指数 logits 并应用掩码
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 计算对数概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(
            1
        )  # 计算对数似然在正样本上的均值

        # loss
        loss = (
            -(self.temperature / self.base_temperature) * mean_log_prob_pos
        )  # 计算损失
        loss = loss.view(anchor_count, batch_size).mean()  # 计算损失均值

        return loss  # 返回损失值
