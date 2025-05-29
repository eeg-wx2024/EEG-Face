from XAI.root_infer import *
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ours.root_path_infer import infer_linear, infer_act, infer_cnn


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, inputs, output):
    if type(inputs[0]) in (list, tuple):
        self.X = []
        for i in inputs[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = inputs[0].detach()
        self.X.requires_grad = True
    self.Y = output


class ExplainFrame(nn.Module):
    def __init__(self):
        super(ExplainFrame, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def explain(self, R):
        return R


# class OriginDTD(ExplainFrame):
#     def explain(self, R):
#         Z = self.forward(self.X)
#         S = safe_divide(R, Z)
#         C = torch.autograd.grad(Z, self.X, S, retain_graph=True)
#
#         if not torch.is_tensor(self.X):
#             outputs = [self.X[0] * C[0], self.X[1] * C[1]]
#         else:
#             outputs = self.X * (C[0])
#         return outputs


def taylor_2nd(Z, X, signal, S, log=''):
    dydx = torch.autograd.grad(Z, X, grad_outputs=torch.ones_like(Z), create_graph=True, retain_graph=True)  # ([],)
    # 2ns gradient
    if dydx[0].requires_grad:
        # print(log, ', 2nd-Taylor Yes!')
        dy2dx = torch.autograd.grad(dydx[0], X, S, retain_graph=True)  # ([],)
    else:
        dy2dx = [0]

    a1 = signal * (dydx[0] * S)
    a2 = torch.pow(signal, exponent=2) * (dy2dx[0]) / 2
    outputs = a1 + a2
    return outputs


class Linear(nn.Linear, ExplainFrame):
    def explain(self, R):
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            _z1 = F.linear(x1, w1)
            _z2 = F.linear(x2, w2)
            _R1 = R * torch.abs(_z1) / (torch.abs(_z1) + torch.abs(_z2))
            _R2 = R * torch.abs(_z2) / (torch.abs(_z1) + torch.abs(_z2))
            S1 = safe_divide(_R1, _z1)
            S2 = safe_divide(_R2, _z2)

            root1 = rel_sup_root_linear(x1, _R1, step=5, weight=w1, z=_z1)
            signal1 = x1 - root1
            R1 = signal1 * torch.autograd.grad(_z1, x1, S1)[0]
            # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear

            root2 = rel_sup_root_linear(x2, _R2, step=10, weight=w2, z=_z2)
            signal2 = x2 - root2
            R2 = signal2 * torch.autograd.grad(_z2, x2, S2)[0]
            # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear

            return R1 + R2

        activator_relevances = f(pw, nw, px, nx)
        # inhibitor_relevances = f(nw, pw, px, nx)
        R = activator_relevances
        return R


''' ----------------------CNNs ---------------------------------'''


class Conv2d(nn.Conv2d, ExplainFrame):

    def explain(self, R):
        stride, padding = self.stride, self.padding
        wp = torch.clamp(self.weight, min=0)

        zp = F.conv2d(self.X, wp, stride=stride, padding=padding)
        # zn = F.conv2d(activation, wn, stride=stride, padding=padding)
        # _R1 = R * torch.abs(zp) / (torch.abs(zp) + torch.abs(zn))
        # _R2 = R * torch.abs(zn) / (torch.abs(zp) + torch.abs(zn))

        root = rel_sup_root_cnn(self.X, R, the_layer=[wp, stride, padding], step=10, z=zp)
        signal = self.X - root

        Sp = safe_divide(R, zp)
        # Sn = safe_divide(_R2, zn)

        R = signal * torch.autograd.grad(zp, self.X, Sp)[0]
        # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear
        return R


class Conv2dInput(nn.Conv2d, ExplainFrame):
    def explain(self, R):
        stride, padding = self.stride, self.padding
        wp = torch.clamp(self.weight, min=0)
        x = torch.ones_like(self.X, dtype=self.X.dtype, requires_grad=True)

        zp = F.conv2d(x, wp, stride=stride, padding=padding)
        # zn = F.conv2d(x, wn, stride=stride, padding=padding)
        # _R1 = R * torch.abs(zp) / (torch.abs(zp) + torch.abs(zn))
        # _R2 = R * torch.abs(zn) / (torch.abs(zp) + torch.abs(zn))

        root = rel_sup_root_cnn(x, R, the_layer=[wp, stride, padding], step=10, z=zp)
        signal = x - root

        Sp = safe_divide(R, zp)
        # Sn = safe_divide(_R2, zn)

        R = signal * torch.autograd.grad(zp, x, Sp)[0]
        # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)  # unnecessary for linear
        # Rn = signal * torch.autograd.grad(zn, x, Sn)[0]
        # f(N) eval
        # self.root_map = root
        return R


class BatchNorm2d(nn.BatchNorm2d, ExplainFrame):
    def explain(self, R_p):
        def f(R, w1, x1):
            Z1 = x1 * w1
            S1 = safe_divide(R, Z1) * w1
            C1 = x1 * S1
            return C1
        def backward(R_p):
            X = self.X

            weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
                (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))

            if torch.is_tensor(self.bias):
                bias = self.bias.unsqueeze(-1).unsqueeze(-1)
                bias_p = safe_divide(bias * R_p.ne(0).type(self.bias.type()),
                                     R_p.ne(0).type(self.bias.type()).sum(dim=[2, 3], keepdim=True))
                R_p = R_p - bias_p

            Rp = f(R_p, weight, X)

            if torch.is_tensor(self.bias):
                Bp = f(bias_p, weight, X)

                Rp = Rp + Bp

            return Rp

        if not torch.is_tensor(R_p):
            idx = len(R_p)
            tmp_R_p = R_p
            Rp = []
            for i in range(idx):
                Rp_tmp = backward(tmp_R_p[i])
                Rp.append(Rp_tmp)
        else:
            Rp = backward(R_p)
        return Rp


class MaxPool2d(nn.MaxPool2d, ExplainFrame):
    def explain(self, R):
        kernel_size, stride, padding = self.kernel_size, self.stride, self.padding
        Z, indices = F.max_pool2d(self.X, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
        Z = Z + 1e-9
        try:
            S = R / Z
            C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_size=self.X.shape)
            R = self.X * C
        except RuntimeError:
            R = R.view(R.size(0), -1, 7, 7)
            S = R / Z
            C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_size=self.X.shape)
            R = self.X * C
        return R
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, ExplainFrame):
    def explain(self, R):
        kernel_size = self.X.shape[-2:]
        Z = F.avg_pool2d(self.X, kernel_size=kernel_size) * kernel_size[0] ** 2 + 1e-9
        S = R / Z
        R = self.X * S
        return R


# class AvgPool2d(nn.AvgPool2d, ExplainFrame):
#     def explain(self, R):
#         kernel_size, stride, padding = self.kernel_size, self.stride, self.padding
#         Z, indices = F.max_pool2d(self.X, kernel_size=kernel_size, stride=stride,
#                                   padding=padding, return_indices=True)
#         Z = Z + 1e-9
#         try:
#             S = R / Z
#             C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride,
#                                padding=padding, output_size=self.X.shape)
#             R = self.X * C
#         except RuntimeError:
#             R = R.view(R.size(0), -1, 7, 7)
#             S = R / Z
#             C = F.max_unpool2d(S, indices, kernel_size=kernel_size, stride=stride,
#                                padding=padding, output_size=self.X.shape)
#             R = self.X * C
#         return R


'''-----------------------------Activations--------------------------------'''


class ReLu(nn.ReLU, ExplainFrame):
    def explain(self, R):
        # xp = torch.clamp(self.X, min=0)
        # xn = torch.clamp(self.X, max=0)
        # Rp = torch.clamp(R, min=0)
        # Rn = torch.clamp(R, max=0)
        # z = F.gelu(self.X)

        # root = rel_sup_root_act(self.X, R, step=20, func=F.gelu, z=z)
        # signal = self.X - root
        # # z1 = F.gelu(xp)
        # # R1 = piece_dtd_act(x=signal1, z=z1, under_R=z1, R=R, root_zero=root1, func=F.gelu, step=50)
        # S = safe_divide(R, z)
        # # R = signal * torch.autograd.grad(z, x, S)[0]
        # R = taylor_2nd(Z=z, X=self.X, signal=signal, S=S)
        return R


class ELU(nn.ELU, ExplainFrame):
    def explain(self, R):
        return R


'''-----------------------------Operations--------------------------------'''


class Dropout(nn.Dropout, ExplainFrame):
    def explain(self, R):
        return R


# class LayerNorm(nn.LayerNorm, ExplainFrame):
#     def explain(self, R):
#         return R


class Add(ExplainFrame):
    def forward(self, inputs):
        return torch.add(*inputs)

    def explain(self, R):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        R0 = self.X[0] * S
        R1 = self.X[1] * S
        return [R0, R1]


class Clone(ExplainFrame):
    def forward(self, x, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(x)
        return outputs

    def explain(self, R_list):
        R = torch.stack(R_list, dim=0)
        R = R.sum(dim=0, keepdim=False)
        return R


# class IndexSelect(ExplainFrame):
#     def forward(self, inputs, dim, indices):
#         self.__setattr__('dim', dim)
#         self.__setattr__('indices', indices)
#         return torch.index_select(inputs, dim, indices)
#
#     def explain(self, R):
#         Z = self.forward(self.X, self.dim, self.indices)
#         S = safe_divide(R, Z)
#         C = torch.autograd.grad(Z, self.X, S, retain_graph=True)
#
#         if not torch.is_tensor(self.X):
#             outputs = [self.X[0] * C[0], self.X[1] * C[1]]
#         else:
#             outputs = self.X * (C[0])
#         return outputs


# class Cat(ExplainFrame):
#     def forward(self, inputs, dim):
#         self.__setattr__('dim', dim)
#         return torch.cat(inputs, dim)
#
#     def explain(self, R):
#         Z = self.forward(self.X, self.dim)
#         S = safe_divide(R, Z)
#         C = self.gradprop(Z, self.X, S)
#
#         outputs = []
#         for x, c in zip(self.X, C):
#             outputs.append(x * c)
#
#         return outputs
#
#
class Sequential(nn.Sequential):
    def explain(self, R):
        for m in reversed(self._modules.values()):
            R = m.explain(R)
        return R

#
# class AddEye(OriginDTD):
#     # input of shape B, C, seq_len, seq_len
#     def forward(self, input):
#         return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)
#
#
# class Einsum(OriginDTD):
#     def __init__(self, equation):
#         super().__init__()
#         self.equation = equation
#
#     def forward(self, *operands):
#         return torch.einsum(self.equation, *operands)
