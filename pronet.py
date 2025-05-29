import torch.nn as nn
import torch
from new_conv import ADCTConv2d
from einops.layers.torch import Rearrange

from new_conv.wtconv2d import WTConv2d

class InceptionConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        k1 = 255
        k2 = 127
        k3 = 63
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,k1), stride=1, padding=(0,k1 // 2), bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,k2), stride=1, padding=(0,k2 // 2), bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,k3), stride=1, padding=(0,k3 // 2), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv2(x) + self.conv3(x)
    
class P5(nn.Module):    # EEGNet
    def __init__(self, spatial=96, temporal=512):
        super().__init__()
        # possible spatial [128, 96, 64, 32, 16, 8]
        # possible temporal [1024, 512, 440, 256, 200, 128, 100, 50]
        F1 = 8
        F2 = 16
        D = 2
        first_kernel = 64
        first_padding = first_kernel // 2
        self.network = nn.Sequential(
            # nn.ZeroPad2d((first_padding, first_padding - 1, 0, 0)),
            # nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, first_kernel)),

            InceptionConv2d(1, F1),####################只改通道
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                in_channels=F1, out_channels=F1, kernel_size=(spatial, 1), groups=F1
            ),
            nn.Conv2d(in_channels=F1, out_channels=D * F1, kernel_size=1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(),
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(
                in_channels=D * F1, out_channels=D * F1, kernel_size=(1, 16), groups=F1
            ),
            nn.Conv2d(in_channels=D * F1, out_channels=F2, kernel_size=1),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(),
        )
        self.fc = nn.Linear(F2 * (temporal // 32), 40)

    def forward(self, x):
        x = self.network(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x),None
    

class A14(nn.Module):
    def __init__(
        self,
        F1=8,
        F2=16,
        D=2,
        nt=512,
        nc=96,
        classes=40,
        dropout=0.25,
    ):
        super().__init__()

        K1 = nt // 2

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((K1 // 2 - 1, K1 // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, K1), bias=False, stride=(1, 1)),
        )
        self.re1 = Rearrange("b f c t -> b c f t")  ###########
        self.bn1 = nn.BatchNorm2d(96)
        self.re2 = Rearrange("b c f t -> b f c t")

        self.conv2 = nn.Conv2d(F1, F1 * D, (nc, 1), bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (nt // 32), classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.re1(x)
        x = self.bn1(x)
        x = self.re2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        # x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        feature = x.squeeze()
        
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x, feature


class P2(nn.Module):
    def __init__(
        self,
        F1=8,
        F2=16,
        D=2,
        nt=512,
        nc=96,
        classes=40,
        dropout=0.5,
    ):
        super().__init__()

        K1 = nt // 2

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((K1 // 2 - 1, K1 // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, K1), bias=False, stride=(1, 1)),
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.re1 = Rearrange("b f c t -> b c f t")###############
        self.adctconv = WTConv2d(126)   #  WTConv2d; ADCTConv2d
        self.re2 = Rearrange("b c f t -> b f c t")

        self.conv2 = nn.Conv2d(F1, F1 * D, (nc, 1), bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        # self.act1 = nn.ELU() ############################################
        self.act1 = nn.Softshrink()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # "ReLU": nn.ReLU(),                0911-1548-21
        # "LeakyReLU": nn.LeakyReLU(0.1),
        # "PReLU": nn.PReLU(),
        # "Sigmoid": nn.Sigmoid(),
        # "Tanh": nn.Tanh(),
        # "Softplus": nn.Softplus(),
        # "SELU": nn.SELU(),
        # "GELU": nn.GELU(),
        # "Hardtanh": nn.Hardtanh(),
        # "ReLU6": nn.ReLU6(),
        # "Softmax": nn.Softmax(dim=0),  # Requires specific dimensionality for Softmax
        # "ELU": nn.ELU(),
        # "Hardshrink": nn.Hardshrink(),
        # "Softshrink": nn.Softshrink(),

        # 时间卷积-1x1卷积------>1x1卷积
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)

        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (nt // 32), classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.re1(x)
        x = self.adctconv(x)
        x = self.re2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x, None
class P1(nn.Module):
    def __init__(
        self,
        F1=8,
        F2=16,
        D=2,
        K2=16,
        nt=512,
        nc=96,
        classes=40,
        dropout=0.25,
    ):
        super().__init__()

        K1 = nt // 2
        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((K1 // 2 - 1, K1 // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, K1), bias=False, stride=(1, 1)),
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.adctconv = ADCTConv2d(F1)

        self.conv2 = nn.Conv2d(F1, F1 * D, (nc, 1), bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((K2 // 2 - 1, K2 // 2, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, K2), bias=False, groups=F1 * D),
        )
        self.conv4 = nn.Conv2d(D * F1, F2, 1)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (nt // 32), classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.adctconv(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.act2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x, None

class P3(nn.Module):

    def __init__(
        self,
        F1=8,
        F2=16,
        D=2,
        K2=16,
        nt=512,
        nc=96,
        classes=40,
        dropout=0.25,
    ):
        super().__init__()

        self.lstm = nn.LSTM(nc, nc, 2, batch_first=True)
        self.net = P2()

    def forward(self, x):
        # x(2, 1, 96, 512)-->(2, 512, 96)

        x = x.squeeze().permute(0, 2, 1)
        # x = x.squeeze()

        x, _ = self.lstm(x)
        # x(2, 512, 96)-->(2, 1, 96, 512)
        x = x.permute(0, 2, 1).unsqueeze(1)
        # x = x.unsqueeze(1)
        x, _ = self.net(x)

        # x(2, 1, 96, 512)-->(2, 1, 96, 512)

        return x, None

class P4(nn.Module):
    def __init__(
        self,
        F1=8,
        F2=16,
        D=2,
        nt=500,
        nc=126,
        classes=40,
        dropout=0.5,
    ):
        super().__init__()

        K1 = nt // 2
        
        self.conv0 = InceptionConv2d(1, F1)####################只改通道
        self.bn0 = nn.BatchNorm2d(F1)

        # self.conv1 = nn.Sequential(
        #     nn.ZeroPad2d((K1 // 2 - 1, K1 // 2, 0, 0)),
        #     nn.Conv2d(1, F1, (1, K1), bias=False, stride=(1, 1)),
        # )
        # self.bn1 = nn.BatchNorm2d(F1)

        self.re1 = Rearrange("b f c t -> b c f t")###############
        self.adctconv = WTConv2d(126)   #  WTConv2d; ADCTConv2d
        self.re2 = Rearrange("b c f t -> b f c t")

        self.conv2 = nn.Conv2d(F1, F1 * D, (nc, 1), bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        # self.act1 = nn.ELU() ############################################
        # self.act1 = nn.Softshrink()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        
        # 时间卷积-1x1卷积------>1x1卷积
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False) # Target layer for Grad-CAM
        
        self.bn3 = nn.BatchNorm2d(F2)
        self.act = nn.ELU() # Activation after target layer
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (nt // 32), classes)

        # Placeholder for activations and gradients
        # self.activation_map = None # Removed: No longer using intermediate activation map for explanation


    def explain(self, x, output, class_index): 
        """
        Generates heatmap based on input gradients.
        Args:
            x: The input tensor that requires gradients.
            output: The model's output tensor (logits).
            class_index: The index of the class for which to generate the heatmap.
                        Can be a single index or a tensor of indices for batch processing.
        Returns:
            heatmap: The generated heatmap (averaged over channels) as a NumPy array.
        """
        if not x.requires_grad:
            raise ValueError("Input tensor 'x' does not require gradients for explanation.")

        # Ensure x retains gradients after backward pass
        x.retain_grad()

        # Handle batch processing for class_index if needed
        if isinstance(class_index, int):
            # If a single index is provided, apply it to all items in the batch
            score = output[:, class_index].sum()
        elif isinstance(class_index, torch.Tensor) and class_index.ndim == 1 and class_index.size(0) == output.size(0):
            # If a tensor of indices (one per batch item) is provided
            score = torch.gather(output, 1, class_index.unsqueeze(-1)).sum()
        else:
            raise ValueError("class_index must be an int or a 1D tensor with size matching the batch size.")

        # Zero previous gradients for x
        if x.grad is not None:
            x.grad.zero_()
            
        # Calculate gradients of the score w.r.t. the input x
        score.backward(retain_graph=True)

        # Get the gradients
        gradients = x.grad
        if gradients is None:
            raise RuntimeError("Could not get gradients for the input tensor x.")

        # Now, we calculate the heatmap for each channel, without averaging
        # Compute the heatmap using the absolute value of gradients, keeping the channel dimension
        heatmap = torch.abs(gradients)  # Shape: (batch, channels, 1, T)

        # Normalize heatmap (per batch item if needed, or globally)
        batch_size = heatmap.size(0)
        heatmap_normalized = torch.zeros_like(heatmap)
        for i in range(batch_size):
            item_heatmap = heatmap[i].squeeze()  # Shape (channels, T)
            min_val = torch.min(item_heatmap)
            max_val = torch.max(item_heatmap)
            if max_val - min_val > 1e-8:
                heatmap_normalized[i] = (heatmap[i] - min_val) / (max_val - min_val + 1e-8)

        # Detach, move to CPU, and convert to NumPy
        heatmap_np = heatmap_normalized.detach().cpu().numpy()  # Shape (batch, channels, T)
        
        # Clean up gradients on x for the next iteration/explanation
        x.grad.zero_() 

        return heatmap_np


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        
        x = self.re1(x)
        x = self.adctconv(x)
        x = self.re2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x) # Original act was here, let's assume it's meant to be ELU

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.bn3(x)
        act_map = self.act(x) # Activation after the target conv layer

        # Store activation map for Grad-CAM
        # self.activation_map = x # Store it before pooling # Removed

        x = self.pool2(act_map) # Pass the activated map to pooling
        x = self.dropout2(x)

        x = self.flatten(x)
        print('x的shape是', x.shape)
        output = self.fc(x)
        
        # No need to set requires_grad here; it depends on the input x
        # output.requires_grad_() # Remove this line

        # Return only the output, not the activation map
        return output #, self.activation_map # Return both output and activation map


class ProNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = P4()   # P1, P2, A14

    def forward(self, x):
        # self.net.forward(x) now returns only the output
        output = self.net(x)
        # Store activation_map temporarily if needed for explain, or rely on P4's internal storage
        # self.activation_map = activation_map # Optional: Store for direct access if needed
        return output #, activation_map # Return only output

    def explain(self, x, output, class_index, class_num=40):
        # Call P4's explain method, passing the input x and output
        return self.net.explain(x, output, class_index)


if __name__ == "__main__":
    model = ProNet()
    # IMPORTANT: Input tensor MUST require grad for Input Gradient explanation
    x = torch.zeros(2, 1, 96, 512, requires_grad=True) # Example batch size 2
    
    # Forward pass now returns only the output
    y = model(x) 
    
    print("Output shape:", y.shape)
    # print("Activation map shape:", act_map.shape) # Removed

    # Example usage of explain (assuming class index 0 for both items in batch)
    # If you have target labels y_true (tensor shape [batch_size]), use that
    example_class_indices = torch.tensor([0, 1]) # Example: explain class 0 for item 0, class 1 for item 1
    if y.size(0) == example_class_indices.size(0):
        try:
            # Pass input x to explain method
            heatmap = model.explain(x, y, class_index=example_class_indices)
            print("Heatmap shape:", heatmap.shape) 
            # Expected heatmap shape: (batch_size, 1, 1, T), e.g., (2, 1, 1, 512)
        except Exception as e:
            print(f"Error during explain call: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
    else:
        print("Batch size mismatch between output and example class indices.")

    # Example with single class index for the whole batch
    try:
        # Re-create x or ensure gradients are zeroed if reusing
        x.grad = None # Clear potential stale gradients before next explain call
        # Ensure requires_grad is still true (it should be unless detach was called)
        if not x.requires_grad: x.requires_grad_(True)

        # Need a new forward pass if gradients from previous backward were cleared or graph was released
        y = model(x) # Re-run forward pass if retain_graph=False was used or gradients were cleared extensively

        heatmap_single_idx = model.explain(x, y, class_index=0)
        print("Heatmap shape (single index):", heatmap_single_idx.shape)
    except Exception as e:
            print(f"Error during explain call (single index): {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
