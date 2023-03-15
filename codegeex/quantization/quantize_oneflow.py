import numpy  as np
import oneflow as torch
from oneflow.nn.parameter import Parameter

def _pack_int8_to_int4(x):
    np_x = x.numpy()
    l = np_x[..., 0::2]
    r = np_x[..., 1::2]
    l = np.left_shift(l, 4)
    if x.dtype is np.int8:
        even = np.bitwise_and(r, np.int8(0xF))
    packed = torch.tensor(np.bitwise_or(l, r), device=x.device)
    return packed


def _quantize(num_bits, symmetric, x, group_dim, group_size, quant_type):
    x_float = x.float()
    x_reshaped = x_float.reshape(
        x.shape[:group_dim]
        + (x.shape[group_dim] // group_size, group_size)
        + x.shape[group_dim + 1 :]
    )
    if symmetric:
        signed_max = float(2 ** (num_bits - 1)) - 1
        offset = signed_max if quant_type is torch.uint8 else 0.0
        scale_float = (
            x_reshaped.abs().max(dim=group_dim + 1, keepdim=True).values / signed_max
        )
        quantized = (
            torch.round(x_reshaped / scale_float + offset)
            .reshape(x.shape)
            .to(quant_type)
        )
        if num_bits == 4:
            quantized = _pack_int8_to_int4(quantized)
        return (quantized, scale_float.squeeze(group_dim + 1).to(x.dtype), None)
    else:
        unsigned_max = float(2 ** num_bits) - 1
        mn = x_reshaped.min(dim=group_dim + 1, keepdim=True).values
        mx = x_reshaped.max(dim=group_dim + 1, keepdim=True).values
        scale_float = (mx - mn) / unsigned_max
        quantized = (
            torch.round((x_reshaped - mn) / scale_float).reshape(x.shape).to(torch.uint8)
        )
        if num_bits == 4:
            quantized = _pack_int8_to_int4(quantized)
        return (
            quantized,
            scale_float.squeeze(group_dim + 1).to(x.dtype),
            mn.squeeze(group_dim + 1).to(x.dtype),
        )

class QuantizedLinear(torch.nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        weight_bit_width: int, 
        weight: torch.Tensor = None, 
        bias: torch.Tensor = None, 
        *args, 
        **kwargs
    ):
        super(QuantizedLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bit_width = weight_bit_width
        self.symmetric = True
        self.group_dim = 1
        self.group_size = in_features

        self.weight, self.weight_scale, self.weight_zero = _quantize(
            self.weight_bit_width, self.symmetric, weight, self.group_dim, self.group_size, torch.int8
        )
        if bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = bias
            self.bias = self.bias.to(kwargs["device"])
        
        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)
        if self.bias is not None:
            self.bias = Parameter(self.bias.to(kwargs["device"]), requires_grad=False)
        if self.weight_zero is not None:
            self.weight_zero = Parameter(self.weight_zero.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Matrix multiply.
        output = torch._C.fused_linear_with_groupwise_quantized_weight(input_, 
                                                                        w=self.weight, 
                                                                        w_scale=self.weight_scale, 
                                                                        w_zero=self.weight_zero, 
                                                                        b=self.bias if self.bias is not None else None, 
                                                                        num_bits=self.weight_bit_width,
                                                                        symmetric=self.symmetric,
                                                                        group_dim=self.group_dim,
                                                                        group_size=self.group_size)
        
        return output

def quantize_oneflow(model, weight_bit_width):
    """Replace fp16 linear with quantized linear"""
    
    for i in range(len(model.language_model.transformer.layers) + 1):
        if i == len(model.language_model.transformer.layers):
            layer = model.language_model.transformer.topQueryLayer
        else:
            layer = model.language_model.transformer.layers[i]
        
        layer.attention.query = QuantizedLinear(
            in_features=layer.attention.query.in_features,
            out_features=layer.attention.query.out_features,
            weight_bit_width=weight_bit_width,
            weight=layer.attention.query.weight.to(torch.cuda.current_device()),
            bias=layer.attention.query.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.query.weight.device,
        )
        layer.attention.value = QuantizedLinear(
            in_features=layer.attention.value.in_features,
            out_features=layer.attention.value.out_features,
            weight_bit_width=weight_bit_width,
            weight=layer.attention.value.weight.to(torch.cuda.current_device()),
            bias=layer.attention.value.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.value.weight.device,
        )
        layer.attention.key = QuantizedLinear(
            in_features=layer.attention.key.in_features,
            out_features=layer.attention.key.out_features,
            weight_bit_width=weight_bit_width,
            weight=layer.attention.key.weight.to(torch.cuda.current_device()),
            bias=layer.attention.key.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.key.weight.device,
        )
        layer.attention.dense = QuantizedLinear(
            in_features=layer.attention.dense.in_features,
            out_features=layer.attention.dense.out_features,
            weight_bit_width=weight_bit_width,
            weight=layer.attention.dense.weight.to(torch.cuda.current_device()),
            bias=layer.attention.dense.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.dense.weight.device,
        )
        layer.mlp.dense_h_to_4h = QuantizedLinear(
            in_features=layer.mlp.dense_h_to_4h.in_features,
            out_features=layer.mlp.dense_h_to_4h.out_features,
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_h_to_4h.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.mlp.dense_h_to_4h.weight.device,
        )
        layer.mlp.dense_4h_to_h = QuantizedLinear(
            in_features=layer.mlp.dense_4h_to_h.in_features,
            out_features=layer.mlp.dense_4h_to_h.out_features,
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_4h_to_h.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.mlp.dense_4h_to_h.weight.device,
        )
        
        
    return model