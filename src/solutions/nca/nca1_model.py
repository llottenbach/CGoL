import torch
import torch.nn as NN
import torch.nn.functional as F

# input, output, hidden -> model -> output, hidden
class NCA1Model(NN.Module):
    def __init__(self, 
                 toroidal=False,
                 hidden_size=100, 
                 perception_size=3, 
                 n_hidden_layers=0, 
                 n_channels=100,
                 activation=NN.ReLU,
                 last_activation=NN.Sigmoid, 
                 device=torch.device('cpu'), 
                 dtype=torch.double,
                 batch_norm=True):
        super().__init__()

        self.is_toroidal = toroidal
        self.perception_size = perception_size
        self.n_hidden_layers = n_hidden_layers
        self.n_channels = n_channels
        self.activation = activation
        self.last_activation = last_activation
        self.activation_fn = activation()
        self.last_activation_fn = last_activation()
        self.device = device
        self.dtype = dtype
        self.batch_norm = batch_norm
        self.hidden_size=hidden_size

        padding = perception_size // 2

        self.padding_mode = 'circular' if toroidal else 'zeros'

        self.conv_start = NN.Conv2d(
            in_channels=2+hidden_size, 
            out_channels=n_channels, 
            kernel_size=perception_size,
            stride=1,
            padding=padding,
            padding_mode=self.padding_mode,
            device=self.device,
            dtype=self.dtype)
        
        if batch_norm:
            self.batch_norm_start = NN.BatchNorm2d(n_channels,
                                                   device=self.device, dtype=self.dtype)
        
        self.hidden_convs = [
            NN.Conv2d(
                in_channels=n_channels, 
                out_channels=n_channels, 
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=self.padding_mode,
                device=self.device,
                dtype=self.dtype) 
            for i in range(n_hidden_layers)]
        self.hidden_batch_norms = [
            NN.BatchNorm2d(n_channels,
                           device=self.device, dtype=self.dtype)
            for i in range(n_hidden_layers)]

        self.conv_end = NN.Conv2d(
            in_channels=n_channels, 
            out_channels=1+hidden_size, 
            kernel_size=1,
            stride=1,
            padding=0,
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
        output = self.conv_start(input)
        output = self.activation_fn(output)
        if self.batch_norm:
            output = self.batch_norm_start(output)
        for hidden_conv, hidden_batch_norm in zip(self.hidden_convs, self.hidden_batch_norms):
            output = hidden_conv(output)
            output = self.activation_fn(output)
            if self.batch_norm:
                output = hidden_batch_norm(output)
        output = self.conv_end(output)
        output = self.last_activation_fn(output)
        return output
    
    def get_config(self) -> dict:
        return {
            "type": type(self).__name__,
            "is_toroidal": self.is_toroidal,
            "perception_size": self.perception_size,
            "activation": type(self.activation).__name__,
            "last_activation": type(self.last_activation).__name__,
            "hidden_size": self.hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "n_channels": self.n_channels,
            "n_parameters": sum([p.numel() for p in self.parameters()]),
            "weight_init": "xavier_uniform",
            "bias_init": "zeros_",
            "batch_norm": self.batch_norm
        }
    
    # batch_size x width x height -> batch_size x (1 + 1 + hidden_size) x width x height
    def init_model_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        return torch.cat((
                input_batch.unsqueeze(-3), 
                torch.zeros((input_batch.shape[0],
                        1 + self.hidden_size,
                        input_batch.shape[1],
                        input_batch.shape[2]),
                    dtype=self.dtype, device=self.device)), 
            dim=-3)
    
    # batch_size x (1 + hidden_size) x width x height -> batch_size x (1 + 1 + hidden_size) x width x height
    def model_input_from_model_output(self, 
                                      model_output: torch.Tensor, 
                                      input_batch: torch.Tensor) -> torch.Tensor:
        return torch.cat((
                input_batch.unsqueeze(-3), 
                model_output), 
            dim=-3)
    
    # batch_size x (1 + hidden_size) x width x height -> batch_size x width x height
    def output_batch_from_model_output(self, 
                                      model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:,0,:,:].squeeze(-3)
    