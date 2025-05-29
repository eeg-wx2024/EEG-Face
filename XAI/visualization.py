import argparse
import torch
import numpy as np
import cv2


def visualize_cam(cam, mean_cropping=True, horizon=1.0):
    # Convert numpy array to torch tensor if necessary
    if isinstance(cam, np.ndarray):
        cam = torch.from_numpy(cam)

    cam = torch.clamp(cam, min=0)
    if torch.isnan(cam).any() or torch.isinf(cam).any():
        print('isnan')
        cam = torch.nan_to_num(cam)

    # Normalize the cam
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize
    if torch.isnan(cam).any() or torch.isinf(cam).any():
        print('isnan')
        cam = torch.nan_to_num(cam)

    if not mean_cropping:
        # Return the heatmap with the full channel dimension (do not squeeze)
        cam = cam.cpu().numpy()  # Return as numpy array without squeezing

    else:
        # Apply cropping mask if needed
        mask = cam.gt(cam.mean() * horizon)
        cam = cam * mask
        cam = cam.cpu().numpy()
    
    return cam  # Return as a 2D array with shape (96, 512)


# def visualize_cam(cam, mean_cropping=True, horizon=1.0):
#     # Convert numpy array to torch tensor if necessary
#     if isinstance(cam, np.ndarray):
#         cam = torch.from_numpy(cam)
        
#     # The input 'cam' after squeeze() in main.py is 1D, shape (W,). 
#     # The original sum was causing the IndexError. We remove it.
#     # cam = torch.sum(cam, dim=1)[0] # Removed this line
    
#     # We should ensure cam is at least 2D for visualization, although it's 1D logically.
#     # Let's keep it 1D for normalization and reshape later if needed by cv2.
    
#     cam = torch.clamp(cam, min=0)
#     if torch.isnan(cam).any() or torch.isinf(cam).any():
#         print('isnan')
#         cam = torch.nan_to_num(cam)
#     cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize
#     # cam = (cam-cam.mean()) / (cam.std())
#     if torch.isnan(cam).any() or torch.isinf(cam).any():
#         print('isnan')
#         cam = torch.nan_to_num(cam)
#     if not mean_cropping:
#         cam = cv2.applyColorMap(np.uint8(cam*255).reshape(1, -1), cv2.COLORMAP_TURBO) # Reshape to (1, W) for cv2
#         cam = cv2.cvtColor(np.array(cam), cv2.COLOR_BGR2RGB)
#     else:
#         # mean cropping serves z+ rule, otherwise it would not see anything
#         mask = cam.gt(cam.mean()*horizon)
#         cam = cam * mask
#         cam = cam.data.cpu().numpy()
#         cam = cv2.applyColorMap(np.uint8(255 * cam).reshape(1, -1), cv2.COLORMAP_TURBO) # Reshape to (1, W) for cv2
#         cam = cv2.cvtColor(np.array(cam), cv2.COLOR_BGR2RGB)

#     # The output is a 2D image, but squeezed to remove the first dimension of size 1
#     return cam.squeeze(0)
