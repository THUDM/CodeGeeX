import torch

from torch.nn.parameter import Parameter
from codegeex.kernels import extract_weight_to_half


class W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, quant_w: torch.Tensor, scale_w: torch.Tensor, weight_bit_width):
        ctx.inp_shape = inp.size()
        ctx.weight_shape = quant_w.size()
        ctx.weight_bit_width = weight_bit_width
        out_features = quant_w.size(0)
        inp = inp.contiguous().view(-1, inp.size(-1))
        weight = extract_weight_to_half(quant_w, scale_w, weight_bit_width)
        output = inp.mm(weight.t())
        ctx.save_for_backward(inp, quant_w, scale_w)
        return output.view(*(ctx.inp_shape[:-1] + (out_features,)))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inp, quant_w, scale_w = ctx.saved_tensors
        weight = extract_weight_to_half(quant_w, scale_w, ctx.weight_bit_width)
        grad_output = grad_output.contiguous().view(-1, weight.size(0))
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(inp)
        return grad_input.view(ctx.inp_shape), grad_weight.view(ctx.weight_shape), None


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

        if weight is None:
            self.weight = torch.empty(
                shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        if bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = bias
        
        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Matrix multiply.
        output = W8A16Linear.apply(input_, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output = output + self.bias
        
        return output


def quantize(model, weight_bit_width):
    """Replace fp16 linear with quantized linear"""
    
    for i in range(len(model.language_model.transformer.layers) + 1):
        if i == len(model.language_model.transformer.layers):
            layer = model.language_model.transformer.topQueryLayer
        else:
            layer = model.language_model.transformer.layers[i]
        
        layer.attention.query = QuantizedLinear(
            in_features=layer.attention.query.weight.shape[0],
            out_features=layer.attention.query.weight.shape[1],
            weight_bit_width=weight_bit_width,
            weight=layer.attention.query.weight.to(torch.cuda.current_device()),
            bias=layer.attention.query.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.query.weight.device,
        )
        layer.attention.value = QuantizedLinear(
            in_features=layer.attention.value.weight.shape[0],
            out_features=layer.attention.value.weight.shape[1],
            weight_bit_width=weight_bit_width,
            weight=layer.attention.value.weight.to(torch.cuda.current_device()),
            bias=layer.attention.value.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.value.weight.device,
        )
        layer.attention.key = QuantizedLinear(
            in_features=layer.attention.key.weight.shape[0],
            out_features=layer.attention.key.weight.shape[1],
            weight_bit_width=weight_bit_width,
            weight=layer.attention.key.weight.to(torch.cuda.current_device()),
            bias=layer.attention.key.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.key.weight.device,
        )
        layer.attention.dense = QuantizedLinear(
            in_features=layer.attention.dense.weight.shape[0],
            out_features=layer.attention.dense.weight.shape[1],
            weight_bit_width=weight_bit_width,
            weight=layer.attention.dense.weight.to(torch.cuda.current_device()),
            bias=layer.attention.dense.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.attention.dense.weight.device,
        )
        layer.mlp.dense_h_to_4h = QuantizedLinear(
            in_features=layer.mlp.dense_h_to_4h.weight.shape[0],
            out_features=layer.mlp.dense_h_to_4h.weight.shape[1],
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_h_to_4h.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.mlp.dense_h_to_4h.weight.device,
        )
        layer.mlp.dense_4h_to_h = QuantizedLinear(
            in_features=layer.mlp.dense_4h_to_h.weight.shape[0],
            out_features=layer.mlp.dense_4h_to_h.weight.shape[1],
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
            bias=layer.mlp.dense_4h_to_h.bias.to(torch.cuda.current_device()),
            params_dtype=torch.half,
            device=layer.mlp.dense_4h_to_h.weight.device,
        )

    return model