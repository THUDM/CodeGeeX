import torch

from torch.nn.parameter import Parameter
from codegeex.kernels import extract_weight_to_half
from codegeex.megatron.mpu.layers import RowParallelLinear, ColumnParallelLinear
from codegeex.megatron.mpu.mappings import copy_to_tensor_model_parallel_region, gather_from_tensor_model_parallel_region, reduce_from_tensor_model_parallel_region, scatter_to_tensor_model_parallel_region


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
                self.out_features, self.in_features * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(self.out_features, dtype=kwargs["params_dtype"], device=kwargs["device"])
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


class QuantizedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_bit_width: int, 
        weight: torch.Tensor = None, 
        bias: torch.Tensor = None, 
        *args, 
        **kwargs,
    ):
        super(QuantizedColumnParallelLinear, self).__init__(input_size, output_size, *args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.weight_bit_width = weight_bit_width
        if "skip_bias_add" in kwargs:
            self.skip_bias_add = kwargs["skip_bias_add"]
        else:
            self.skip_bias_add = False
        del self.weight

        if weight is None:
            self.weight = torch.empty(
                self.output_size, self.input_size * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(self.output_size, dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        if bias is None:
            self.register_parameter('bias', None)
        else:
            del self.bias
            self.bias = bias
            
        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None and not self.skip_bias_add:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
            
        output_bias = self.bias if self.skip_bias_add else None
        
        return output, output_bias


class QuantizedRowParallelLinear(RowParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_bit_width: int, 
        weight: torch.Tensor = None, 
        bias: torch.Tensor = None,  
        *args, 
        **kwargs,
    ):
        super(QuantizedRowParallelLinear, self).__init__(input_size, output_size, *args, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.weight_bit_width = weight_bit_width
        if "skip_bias_add" in kwargs:
            self.skip_bias_add = kwargs["skip_bias_add"]
        else:
            self.skip_bias_add = False
        del self.weight
        
        if weight is None:
            self.weight = torch.empty(
                self.output_size, self.input_size * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(self.output_size, dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        if bias is None:
            self.register_parameter('bias', None)
        else:
            del self.bias
            self.bias = bias
            
        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if self.bias is not None and not self.skip_bias_add:
            output = output_ + self.bias
        else:
            output = output_
        output_bias = self.bias if self.skip_bias_add else None
        
        return output, output_bias
    

def quantize(model, weight_bit_width, backend="torch"):
    """Replace fp16 linear with quantized linear"""
    
    for i in range(len(model.language_model.transformer.layers) + 1):
        if i == len(model.language_model.transformer.layers):
            layer = model.language_model.transformer.topQueryLayer
        else:
            layer = model.language_model.transformer.layers[i]
        
        if backend == "torch":
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
        elif backend == "megatron":
            layer.attention.query = QuantizedColumnParallelLinear(
                weight_bit_width=weight_bit_width,
                weight=layer.attention.query.weight.to(torch.cuda.current_device()),
                bias=layer.attention.query.bias.to(torch.cuda.current_device()),
                input_size=layer.attention.query.input_size,
                output_size=layer.attention.query.output_size,
                gather_output=False,
                skip_init=True,
                params_dtype=torch.half,
                device=layer.attention.query.weight.device,
            )
            layer.attention.value = QuantizedColumnParallelLinear(
                weight_bit_width=weight_bit_width,
                weight=layer.attention.value.weight.to(torch.cuda.current_device()),
                bias=layer.attention.value.bias.to(torch.cuda.current_device()),
                input_size=layer.attention.value.input_size,
                output_size=layer.attention.value.output_size,
                gather_output=False,
                skip_init=True,
                params_dtype=torch.half,
                device=layer.attention.value.weight.device,
            )
            layer.attention.key = QuantizedColumnParallelLinear(
                weight_bit_width=weight_bit_width,
                weight=layer.attention.key.weight.to(torch.cuda.current_device()),
                bias=layer.attention.key.bias.to(torch.cuda.current_device()),
                input_size=layer.attention.key.input_size,
                output_size=layer.attention.key.output_size,
                gather_output=False,
                skip_init=True,
                params_dtype=torch.half,
                device=layer.attention.key.weight.device,
            )
            layer.attention.dense = QuantizedRowParallelLinear(
                weight_bit_width=weight_bit_width,
                weight=layer.attention.dense.weight.to(torch.cuda.current_device()),
                bias=layer.attention.dense.bias.to(torch.cuda.current_device()),
                input_size=layer.attention.dense.input_size,
                output_size=layer.attention.dense.output_size,
                input_is_parallel=False,
                skip_init=True,
                skip_bias_add=True,
                params_dtype=torch.half,
                device=layer.attention.dense.weight.device,
            )
            layer.mlp.dense_h_to_4h = QuantizedColumnParallelLinear(
                weight_bit_width=weight_bit_width,
                weight=layer.mlp.dense_h_to_4h.weight.to(torch.cuda.current_device()),
                bias=layer.mlp.dense_h_to_4h.bias.to(torch.cuda.current_device()),
                input_size=layer.mlp.dense_h_to_4h.input_size,
                output_size=layer.mlp.dense_h_to_4h.output_size,
                gather_output=False,
                skip_init=True,
                params_dtype=torch.half,
                device=layer.mlp.dense_h_to_4h.weight.device,
            )
            layer.mlp.dense_4h_to_h = QuantizedRowParallelLinear(
                weight_bit_width=weight_bit_width,
                weight=layer.mlp.dense_4h_to_h.weight.to(torch.cuda.current_device()),
                bias=layer.mlp.dense_4h_to_h.bias.to(torch.cuda.current_device()),
                input_size=layer.mlp.dense_4h_to_h.input_size,
                output_size=layer.mlp.dense_4h_to_h.output_size,
                input_is_parallel=False,
                skip_init=True,
                params_dtype=torch.half,
                device=layer.mlp.dense_4h_to_h.weight.device,
            )
            
    return model