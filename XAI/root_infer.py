# -*- coding: utf-8 -*-
"""
 @time: 2024/1/17 10:36
 @desc:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
import einops


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)  # set the min bound, means get larger than 1e-9, the "stabilizer"
    den = den + den.eq(0).type(den.type()) * 1e-9  # if den==0 then +1*1e-9
    return a / den * b.ne(0).type(b.type())  # do / if b=!0 or *0


def rel_sup_root_linear(x, R, step=50, weight=None, z=None):
    lr = 1
    signal_new = x.clone().detach().requires_grad_(True)
    # x_new = torch.nn.Parameter(x_new, requires_grad=True)
    # optimizer = torch.optim.SGD([x_new], lr=1)
    # grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), retain_graph=True)[0]

    # R_ = F.softmax(R, dim=-1)
    R_ = R
    for _ in range(step):
        # x_new = torch.relu(x_new)
        y = F.linear(signal_new, weight=weight)
        # y = F.softmax(y, dim=-1)
        # gradient-based gamma
        # noise = x - signal_new
        # y_n = F.linear(noise, weight=weight)
        # grad_n = torch.autograd.grad(outputs=y_n, inputs=noise, grad_outputs=torch.ones_like(y_n))[0]

        # loss = torch.pow(y - R, 2)
        # loss = nn.functional.cross_entropy(y, R)
        # loss = nn.functional.l1_loss(y, R)
        # loss = nn.functional.mse_loss(y, R)
        # loss = nn.functional.poisson_nll_loss(y, R)
        # loss = nn.functional.kl_div(y, R, reduce=True, size_average=False) / 16
        loss = nn.functional.hinge_embedding_loss(y, R_)
        # loss = nn.functional.smooth_l1_loss(y, R)
        # loss.backward(retain_graph=True)
        # optimizer.step

        # gradient-based gamma
        # temp_dif = (torch.abs(grad_x - grad_n) - 0.15) * 10
        # ad = torch.pow(torch.ones_like(x) / 10, torch.where(temp_dif > 0, temp_dif, 0))
        # lr = lr * ad

        grad_interp = torch.autograd.grad(outputs=loss, inputs=signal_new, grad_outputs=torch.ones_like(loss))[0]
        grad_interp = torch.clamp(grad_interp, max=1)
        delta = lr * grad_interp
        signal_new = signal_new - delta

    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    # x_new = torch.relu(x_new)
    # scale_s = x_new.mean()
    # x_new = x_new * scale_x / scale_s
    root = x - signal_new
    root = root.detach()
    return root


def rel_sup_root_act(x, R, z, step=50, func=None):
    # signal_new = x.clone().detach().requires_grad_(True)
    # gradient-based gamma
    signal_new = torch.zeros_like(x).requires_grad_(True) + 10e-3
    grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), retain_graph=True)[0]
    # lr = 3 * torch.ones_like(signal_new).to(signal_new)
    lr = 30

    # x_new = torch.nn.Parameter(x_new, requires_grad=True)
    # optimizer = torch.optim.SGD([x_new], lr=1)
    # FC layer for output
    # R_ = F.softmax(R, dim=-1)
    R_ = R
    for _ in range(step):
        # linear constrain

        if func is F.softmax:
            y = func(signal_new, dim=-1)

            # gradient-based gamma
            noise = x - signal_new
            y_n = func(noise, dim=-1)
            grad_n = torch.autograd.grad(outputs=y_n, inputs=noise, grad_outputs=torch.ones_like(y_n))[0]
        else:
            y = func(signal_new)
            # y = F.softmax(y, dim=-1)

            # gradient-based gamma
            noise = x - signal_new
            y_n = func(noise)
            grad_n = torch.autograd.grad(outputs=y_n, inputs=noise, grad_outputs=torch.ones_like(y_n))[0]

        # loss = torch.pow(y - R, 2)
        # loss = nn.functional.l1_loss(y, R)
        # loss = nn.functional.mse_loss(y, R)
        # loss = nn.functional.poisson_nll_loss(y, R)
        # loss = nn.functional.kl_div(y, R, reduce=True, size_average=False) / 16
        # print(x.shape, y.shape, R_.shape, '1')
        loss = nn.functional.hinge_embedding_loss(y, R_)
        # loss = nn.functional.smooth_l1_loss(y, R)
        # loss = nn.functional.cross_entropy(y, R)

        # lr = lr * torch.clamp(ad, max=1)
        # gradient-based gamma
        # ad = torch.ones_like(x) / torch.pow(torch.ones_like(x), exponent=(torch.abs(grad_x - grad_n) - 0.15)*1.2)
        # temp_0 = torch.zeros_like(grad_x)
        temp_dif = (torch.abs(grad_x - grad_n) - 0.15) * 10
        ad = torch.pow(torch.ones_like(x) / 10, torch.where(temp_dif > 0, temp_dif, 0))
        lr = lr * ad

        # loss.backward(retain_graph=True)
        # optimizer.step()
        grad_interp = torch.autograd.grad(outputs=loss, inputs=signal_new, grad_outputs=torch.ones_like(loss))[0]
        # grad_interp = torch.clamp(grad_interp, max=1)
        delta = lr * grad_interp
        signal_new = signal_new - delta

    # 此时可以假设认为 f(x_new)=class response, f(root)=0
    root = x - signal_new
    root = root.detach()
    return root


def rel_sup_root_cnn(x, R, step=50, the_layer=None, z=None):
    lr = 1
    signal_new = x.clone().detach().requires_grad_(True)
    # grad_x = torch.autograd.grad(outputs=z, inputs=x, grad_outputs=torch.ones_like(z), retain_graph=True)[0]

    w = R.shape[-1]
    # R_ = einops.rearrange(R, 'b c h w -> b c (h w)')
    # R_ = F.softmax(R_, dim=-1)
    # R_ = einops.rearrange(R_, 'b c (h w) -> b c h w', w=w)
    R_ = R
    for _ in range(step):
        # x_new = torch.relu(x_new)
        y = F.conv2d(signal_new, weight=the_layer[0], stride=the_layer[1], padding=the_layer[2])

        # gradient-based gamma
        # noise = x - signal_new
        # y_n = F.conv2d(noise, weight=the_layer[0], stride=the_layer[1], padding=the_layer[2])
        # grad_n = torch.autograd.grad(outputs=y_n, inputs=noise, grad_outputs=torch.ones_like(y_n))[0]

        # y = einops.rearrange(y, 'b c h w -> b c (h w)')
        # y = F.softmax(y, dim=-1)
        # y = einops.rearrange(y, 'b c (h w) -> b c h w', w=w)
        loss = pytorch_ssim.ssim(y, R_)

        # gradient-based gamma
        # temp_dif = (torch.abs(grad_x - grad_n) - 0.15) * 10
        # ad = torch.pow(torch.ones_like(x) / 10, torch.where(temp_dif > 0, temp_dif, 0))
        # lr = lr * ad

        grad_interp = torch.autograd.grad(outputs=loss, inputs=signal_new, grad_outputs=torch.ones_like(loss))[0]
        # grad_interp = torch.clamp(grad_interp, max=1)
        delta = lr * grad_interp
        signal_new = signal_new - delta

    # x_new = torch.relu(x_new)
    root = x - signal_new
    root = root.detach()
    return root


# def rel_sup_root_act2d(x, R, step=50, func=None):
#     x_new = x.clone()
#     lr = 0.01
#     # x_new = torch.nn.Parameter(x_new, requires_grad=True)
#     # optimizer = torch.optim.SGD([x_new], lr=alpha)
#     # FC layer for output
#     for _ in range(step):
#         if func is F.softmax:
#             y = func(x_new, dim=-1)
#         else:
#             y = func(x_new)
#
#         # R = F.softmax(R, dim=-1)
#         loss = nn.functional.cross_entropy(y, R)
#         # loss.backward(retain_graph=True)
#         # optimizer.step()
#         # loss = torch.pow(y - R, 2)
#         grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
#         delta = lr * grad_interp
#         x_new = x_new - delta
#     # 此时可以假设认为 f(x_new)=class response, f(root)=0
#     root = x - x_new
#     root = root.detach()
#     return root


def rel_sup_root_linear_v2(x, R, step=20, func=None):
    lr = 1
    x_new = x.clone().detach().requires_grad_(True)

    R_ = F.softmax(R, dim=-1)
    for _ in range(step):
        y = func(x_new)
        y = F.softmax(y, dim=-1)
        loss = nn.functional.hinge_embedding_loss(y, R_)
        grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
        grad_interp = torch.clamp(grad_interp, max=1)
        delta = lr * grad_interp
        x_new = x_new - delta

    root = x - x_new
    root = root.detach()
    return root


def rel_sup_root_cnn_v2(x, R, step=20, func=None):
    lr = 1
    x_new = x.clone().detach().requires_grad_(True)

    w = R.shape[-1]
    R_ = einops.rearrange(R, 'b c h w -> b c (h w)')
    R_ = F.softmax(R_, dim=-1)
    R_ = einops.rearrange(R_, 'b c (h w) -> b c h w', w=w)
    for _ in range(step):
        y = func(x_new)
        y = einops.rearrange(y, 'b c h w -> b c (h w)')
        y = F.softmax(y, dim=-1)
        y = einops.rearrange(y, 'b c (h w) -> b c h w', w=w)
        loss = pytorch_ssim.ssim(y, R_)
        grad_interp = torch.autograd.grad(outputs=loss, inputs=x_new, grad_outputs=torch.ones_like(loss))[0]
        grad_interp = torch.clamp(grad_interp, max=1)
        delta = lr * grad_interp
        x_new = x_new - delta

    root = x - x_new
    root = root.detach()
    return root
