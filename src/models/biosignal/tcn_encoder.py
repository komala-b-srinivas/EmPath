import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2)
        self.activation = nn.ReLU() 
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)  # Skip connection

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.activation(out)
        skip = self.skip_conv(residual)  # Skip connection
        out += skip
        return out


def causal_conv1d(in_channels, out_channels, kernel_size, dilation):
    return nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,
                     padding=(kernel_size - 1) * dilation)


class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_channels, num_blocks, num_classes, kernel_size=3, dilation_factor=2):
        super(TemporalConvolutionalNetwork, self).__init__()
        layers = []
        in_channels = input_channels
        for i in range(num_blocks):
            dilation = dilation_factor ** i
            out_channels = 64  # You can adjust this as needed
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, dilation))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.network(x)
        out = out.mean(dim=2)  # Global average pooling
        out = self.fc(out)  # Fully connected layer
        return out


class TCNBiosignalEncoder(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TCNBiosignalEncoder, self).__init__()
        self.tcn = TemporalConvolutionalNetwork(input_channels, num_blocks=4, num_classes=num_classes)

    def forward(self, x):
        return self.tcn(x)  # Calls the TCN forward method