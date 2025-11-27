import torch
import torch.nn as NN
import torch.nn.functional as F

# 10layers x 3kernelsize = 21 width
#  5layers x 5kernelsize = 21 width
#  4layers x 7kernelsize = 25 width
#  3layers x 9kernelsize = 25 width

class Conv1Model(NN.Module):
    def __init__(self, 
                 toroidal=False, 
                 kernel_size=5, 
                 n_hidden_layers=3, 
                 n_channels=100,
                 activation=NN.ReLU,
                 last_activation=NN.Sigmoid, 
                 device=torch.device('cpu'), 
                 dtype=torch.double):
        super().__init__()

        self.is_toroidal = toroidal
        self.kernel_size = kernel_size
        self.n_hidden_layers = n_hidden_layers
        self.n_channels = n_channels
        self.activation = activation
        self.last_activation = last_activation
        self.activation_fn = activation()
        self.last_activation_fn = last_activation()
        self.device = device
        self.dtype = dtype

        padding = kernel_size // 2

        self.padding_mode = 'circular' if toroidal else 'zeros'

        self.conv_start = NN.Conv2d(
            in_channels=1, 
            out_channels=n_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype)
        
        self.hidden_convs = [
            NN.Conv2d(
                in_channels=n_channels, 
                out_channels=n_channels, 
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=self.padding_mode,
                device=self.device,
                dtype=self.dtype) 
            for i in range(n_hidden_layers)]

        self.conv_end = NN.Conv2d(
            in_channels=n_channels, 
            out_channels=1, 
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype)
        
    def initialize(self):
        init_weight_f = NN.init.xavier_uniform
        init_bias_f = NN.init.zeros_

        init_weight_f(self.conv_start.weight)
        init_bias_f(self.conv_start.bias)
        for hidden_conf in self.hidden_convs:
            init_weight_f(hidden_conf.weight)
            init_bias_f(hidden_conf.bias)
        init_weight_f(self.conv_end.weight)
        init_bias_f(self.conv_end.bias)

    def forward(self, input: torch.Tensor):
        output = input.unsqueeze(-3)
        output = self.conv_start(output)
        output = self.activation_fn(output)
        for hidden_conv in self.hidden_convs:
            output = hidden_conv(output)
            output = self.activation_fn(output)
        output = self.conv_end(output)
        output = self.last_activation_fn(output)
        return output.squeeze(-3)
    
    def get_config(self) -> dict:
        return {
            "type": type(self).__name__,
            "is_toroidal": self.is_toroidal,
            "kernel_size": self.kernel_size,
            "activation": type(self.activation).__name__,
            "last_activation": type(self.last_activation).__name__,
            "n_hidden_layers": self.n_hidden_layers,
            "n_channels": self.n_channels,
            "n_parameters": sum([p.numel() for p in self.parameters()]),
            "weight_init": "xavier_uniform",
            "bias_init": "zeros_"
        }
    