# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
layer_norm_x_backprop_v2
"""
import operator

import impl.layer_norm_x_backprop_v2_unify as layer_norm_x_backprop_v2_unify
import te.lang.cce as tbe
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe_context
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from te import platform as tbe_platform
from te import tvm
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector


# 'pylint: disable=too-many-lines
# 'pylint: disable = unused-argument,too-many-arguments,too-many-locals,global-variable-undefined
def get_op_support_info(input_dy,
                        input_x,
                        input_variance,
                        input_mean,
                        input_gamma,
                        output_pd_x,
                        res_for_gamma,
                        kernel_name="layer_norm_x_backprop_v2"):
    """
    get_op_support_info
    """
    shape_x = input_x.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")
    format_dy = input_dy.get("format").upper()
    if format_dy in ("ND", "NCHW", "NHWC", "NC1HWC0"):
        if len(shape_x) == len(shape_gamma):
            axis_split_matrix = []
            flag = -1
            for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
                if xtem != mean:
                    flag = i
                    break
            if flag == -1:
                for i in range(len(shape_x) - 1):
                    split_0 = [
                        SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]], [2, [i], [-1], [-1]],
                                   [3, [i], [-1], [-1]], [4, [i], [-1], [-1]]),
                        SplitOutput([0, [i]])
                    ]
                    axis_split_matrix.append(split_0)
            else:
                for i in range(flag):
                    split_0 = [
                        SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]], [2, [i], [-1], [-1]],
                                   [3, [i], [-1], [-1]], [4, [i], [-1], [-1]]),
                        SplitOutput([0, [i]], [1, [i]])
                    ]
                    axis_split_matrix.append(split_0)
        else:
            axis_split_matrix = None

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=invalid-name,too-many-lines,too-many-arguments
# 'pylint: disable=unused-argument,too-many-locals,locally-disabled
def _check_dynamic_format(shape_dy, shape_gamma, c_0):
    """
    check dynamic format branch

    """
    if len(shape_dy) < 2 or len(shape_gamma) != 1:
        return True
    if shape_dy[-1] % c_0 != 0 or shape_dy[-2] % c_0 != 0 \
            or shape_gamma[-1] % c_0 != 0:
        return True
    return True


def op_select_format(input_dy,
                     input_x,
                     input_variance,
                     input_mean,
                     input_gamma,
                     output_pd_x,
                     res_for_gamma,
                     kernel_name="layer_norm_x_backprop_v2"):
    """
    function of selecting dynamic format

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    None
    """
    shape_dy = input_dy.get("ori_shape")
    shape_gamma = input_gamma.get("ori_shape")
    shape_dy = shape_util.scalar2tensor_one(shape_dy)
    shape_gamma = shape_util.scalar2tensor_one(shape_gamma)
    c_0 = 16

    if _check_dynamic_format(shape_dy, shape_gamma, c_0):
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="dy",
                                               datatype="float16,float16,float16,float,"
                                                        "float,float",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16,float16,float16,float,"
                                                        "float,float",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="variance",
                                               datatype="float16,float16,float16,float,"
                                                        "float,float",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="mean",
                                               datatype="float16,float16,float16,float,"
                                                        "float,float",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="gamma",
                                               datatype="float16,float16,float16,float,"
                                                        "float,float",
                                               format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="pd_x",
                                                datatype="float16,float16,float16,float,"
                                                         "float,float",
                                                format="NCHW,NHWC,ND,NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="res_for_gamma",
                                                datatype="float,float,float,float,"
                                                         "float,float",
                                                format="NCHW,NHWC,ND,NCHW,NHWC,ND")
    else:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="dy",
                                               datatype="float16, float,float16,float16,"
                                                        "float16,float,float,float",
                                               format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        input1 = util_select_op_base.gen_param(classify="input1",
                                               name="x",
                                               datatype="float16, float,float16,float16,"
                                                        "float16,float,float,float",
                                               format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,ND,"
                                                      "NCHW,NHWC,ND")
        input2 = util_select_op_base.gen_param(classify="input2",
                                               name="variance",
                                               datatype="float16, float,float16,float16,"
                                                        "float16,float,float,float",
                                               format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                      "NHWC,ND")
        input3 = util_select_op_base.gen_param(classify="input3",
                                               name="mean",
                                               datatype="float16, float,float16,float16,"
                                                        "float16,float,float,float",
                                               format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                      "NHWC,ND")
        input4 = util_select_op_base.gen_param(classify="input4",
                                               name="gamma",
                                               datatype="float16, float,float16,float16,"
                                                        "float16,float,float,float",
                                               format="ND,ND,NCHW,NHWC,ND,NCHW,"
                                                      "NHWC,ND")
        output0 = util_select_op_base.gen_param(classify="output0",
                                                name="pd_x",
                                                datatype="float16, float,float16,float16,"
                                                         "float16,float,float,float",
                                                format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                       "ND,NCHW,NHWC,ND")
        output1 = util_select_op_base.gen_param(classify="output1",
                                                name="res_for_gamma",
                                                datatype="float, float,float,float,"
                                                         "float,float,float,float",
                                                format="FRACTAL_NZ,FRACTAL_NZ,NCHW,NHWC,"
                                                       "ND,NCHW,NHWC,ND")

    param_list = [input0, input1, input2, input3, input4, output0, output1]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_params(params_map):
    """
    check parameters including shape_dy, shape_x, shape_var,
    shape_mean, shape_gamma, dtype and kernel_name

    Parameters
    ----------
    params_map: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
        "shape_mean": shape_mean, "shape_gamma": shape_gamma,
        "dtype": dtype, "kernel_name": kernel_name}

    Returns
    -------
    None
    """

    check_list = ("float16", "float32")
    para_check.check_dtype(params_map.get("dtype"), check_list, param_name="input_dy")

    _check_shape(params_map)


def _check_shape(params_map):
    """
    check parameters including shape_dy, shape_x, shape_var,
    shape_mean and shape_gamma

    Parameters
    ----------
    params_map: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma,
         "dtype": dtype, "kernel_name": kernel_name}

    Returns
    -------
    None
    """
    if operator.ne(tuple(params_map.get("shape_dy")), tuple(params_map.get("shape_x"))):
        error_detail = "shape of input_dy and input_x should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_x_backprop_v2", "input_dy", "input_x",
                                                               error_detail)

    if operator.ne(tuple(params_map.get("shape_var")), tuple(params_map.get("shape_mean"))):
        error_detail = "shape of input_variance and input_mean should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_x_backprop_v2", "input_variance",
                                                               "input_mean", error_detail)

    shape_x = params_map.get("shape_x")
    shape_mean = params_map.get("shape_mean")
    shape_gamma = params_map.get("shape_gamma")

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_mean, param_name="input_mean")
    para_check.check_shape(shape_gamma, param_name="input_gamma")

    _check_shape_mean(shape_x, shape_mean)
    _check_shape_gamma(shape_x, shape_gamma)


def _check_shape_mean(shape_x, shape_mean):
    """
    check if parameter shape_mean meets the requirements of function

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean

    Returns
    -------
    None
    """
    if len(shape_x) != len(shape_mean):
        error_detail = "length of shape_x and shape_mean should be same"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_x_backprop_v2", "input_x", "input_mean",
                                                               error_detail)

    if shape_mean[-1] != 1:
        error_detail = "value of shape_mean's last dim must be 1"
        error_manager_vector.raise_err_input_shape_invalid("layer_norm_x_backprop_v2", "input_mean", error_detail)

    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break

    if flag != -1:
        for i, mean in enumerate(shape_mean):
            if i < flag:
                continue
            if mean != 1:
                error_detail = "value of shape_mean must be 1"
                error_manager_vector.raise_err_input_shape_invalid("layer_norm_x_backprop_v2", "input_mean",
                                                                   error_detail)


def _check_shape_gamma(shape_x, shape_gamma):
    """
    check if parameter shape_gamma meets the requirements of function

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    None
    """
    if len(shape_gamma) > len(shape_x):
        error_detail = "length of shape_gamma can not be longer than shape_x"
        error_manager_vector.raise_err_two_input_shape_invalid("layer_norm_x_backprop_v2", "input_gamma", "input_x",
                                                               error_detail)

    for xtem, gamma in zip(reversed(shape_x), reversed(shape_gamma)):
        if xtem != gamma:
            error_detail = "value of shape_gamma is wrong"
            error_manager_vector.raise_err_input_shape_invalid("layer_norm_x_backprop_v2", "input_gamma", error_detail)


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


def _update_gamma_shape(shape_x, shape_gamma):
    """
    update shape_gamma for subsequent calculation

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    shape_gamma_new: tuple
        new shape_gamma after update
    params_axis: tuple
        the list of axis for gamma reduce_sum
    """
    params_axis_tmp = []
    if len(shape_x) != len(shape_gamma):
        sub = len(shape_x) - len(shape_gamma)
        shape_gamma = list(shape_gamma)
        for i in range(sub):
            shape_gamma.insert(0, 1)
            params_axis_tmp.append(i)

    shape_gamma_new = tuple(shape_gamma)
    params_axis = tuple(params_axis_tmp)

    return shape_gamma_new, params_axis


def _get_data_gm(shapes, dtype):
    """
    get placeholders of data_dy, data_x, data_variance, data_mean and data_gamma

    Parameters
    ----------
    shapes: dict
        {"shape_dy": shape_dy, "shape_x": shape_x, "shape_var": shape_variance,
         "shape_mean": shape_mean, "shape_gamma": shape_gamma}
    dtype: str
        the data type

    Returns
    -------
    data_gm: tuple
        (data_dy, data_x, data_variance, data_mean, data_gamma)
    """
    data_dy = tvm.placeholder(shapes.get("shape_dy"), name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(shapes.get("shape_x"), name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(shapes.get("shape_var"), name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(shapes.get("shape_mean"), name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(shapes.get("shape_gamma"), name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def _get_params(shape_x, shape_mean, shape_gamma):
    """
    compute parameters including param_axis, reduce_axis and mean_num

    Parameters
    ----------
    shape_x: list or tuple
        shape of x
    shape_mean: list or tuple
        shape of mean
    shape_gamma: list or tuple
        shape of gamma

    Returns
    -------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    """
    param_axis = _update_gamma_shape(shape_x, shape_gamma)[1]

    reduce_axis_tmp = []
    flag = -1
    for i, (xtem, mean) in enumerate(zip(shape_x, shape_mean)):
        if xtem != mean:
            flag = i
            break
    if flag != -1:
        for i in range(flag, len(shape_x)):
            reduce_axis_tmp.append(i)
    else:
        reduce_axis_tmp.append(len(shape_x) - 1)
    reduce_axis = tuple(reduce_axis_tmp)

    mean_num = 1.0
    for i in reduce_axis:
        mean_num *= shape_x[i]

    params = {"param_axis": param_axis, "reduce_axis": reduce_axis, "mean_num": mean_num}

    return params


def _get_pd_xl(data, shape_x):
    """
    compute pd_xl according to data_dy, data_gamma and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    shape_x: list or tuple
        shape of x

    Returns
    -------
    pd_xl: tvm.tensor
        data_dy*data_gamma
    """
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), shape_x)
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))

    return pd_xl


def _get_pd_var_front(data, cast_dtype):
    """
    compute front part of pd_var according to data_variance

    Parameters
    ----------
    data: dict
        placeholders after cast
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_var_1: tvm.tensor
        np.power((data_variance + EPSLON), (-1.5))
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    """
    var_elta = tbe.vadds(data.get("data_variance"), tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log, tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)
    pdvar1_mul = tbe.vmul(var_elta_2, var_elta_2)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_2)

    return pd_var_1, var_elta_2


def _get_pd_var(data, params, shape_x, pd_xl, cast_dtype):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    pd_xl: tvm.tensor
        data_dy*data_gamma
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))), reduce_axis,
        keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    """
    pd_var_1, var_elta_2 = _get_pd_var_front(data, cast_dtype)

    data_mean_cast = tbe.broadcast(data.get("data_mean"), shape_x)
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = tbe.vmul(sub_x_mean, pd_xl)
    pdvar_sum = tbe.sum(pdvar_mul1, params.get("reduce_axis"), keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    return pd_var, var_elta_2, sub_x_mean


def _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype):
    """
    compute pd_mean according to reduce_axis, pd_xl, pd_var, var_elta_2
    and sub_x_mean

    Parameters
    ----------
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    pd_xl: tvm.tensor
        data_dy*data_gamma
    pd_var: tvm.tensor
        np.sum(((-0.5)*pd_xl*(data_x - data_mean)
        *np.power((data_variance + EPSLON), (-1.5))), reduce_axis,
        keepdims=True)
    var_elta_2: tvm.tensor
        np.power((data_variance + EPSLON), (-0.5))
    sub_x_mean: tvm.tensor
        data_x - data_mean
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_mean: tvm.tensor
        np.sum(((-1.0)*pd_xl
        *np.power((data_variance + EPSLON), (-0.5))), reduce_axis,
        keepdims=True)
        + pd_var*(1.0/m)*np.sum(((-2.0)*(data_x - data_mean)),
        reduce_axis, keepdims=True)
    """
    pdmean1_sum = tbe.sum(pd_xl, params.get("reduce_axis"), keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean_1 = tbe.vmuls(pdmean1_mul, tvm.const(-1.0, dtype=cast_dtype))

    return pd_mean_1


def _get_pd_x(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x, res_for_gamma according to data, params and shape_x

    `pd_x     = pd_xl * np.power((data_variance + EPSLON), (-0.5))`
               `+ pd_var * (2.0 / m) * (data_x - data_mean) + pd_mean * (1.0 / m)`
    `res_for_gamma = (data_x - data_mean) * np.power((data_variance + EPSLON), (-0.5))`

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    res_for_gamma: tvm.tensor
        `(data_x - data_mean)*np.power((data_variance + EPSLON), (-0.5))`
    """
    pd_xl = _get_pd_xl(data, shape_x)

    pd_var, var_elta_2, sub_x_mean = _get_pd_var(data, params, shape_x, pd_xl, cast_dtype)

    pd_mean = _get_pd_mean(params, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype)

    var_elta_2_cast = tbe.broadcast(var_elta_2, shape_x)
    pd_x_1 = tbe.vmul(var_elta_2_cast, pd_xl)
    res_for_gamma = tbe.vmul(var_elta_2_cast, sub_x_mean)

    pd_var = tbe.vmuls(pd_var, tvm.const((2 * (params.get("mean_num") ** (-1))), dtype=cast_dtype))
    pdx2_broad = tbe.broadcast(pd_var, shape_x)
    pd_x_2 = tbe.vmul(pdx2_broad, sub_x_mean)
    pd_x_3 = tbe.vmuls(pd_mean, tvm.const((params.get("mean_num") ** (-1)), dtype=cast_dtype))

    pdx_broad = tbe.broadcast(pd_x_3, shape_x)
    pdx_add = tbe.vadd(pd_x_1, pd_x_2)
    pd_x_ub = tbe.vadd(pdx_add, pdx_broad)

    if dtype == "float16" and cast_dtype == "float32":
        pd_x = tbe.cast_to(pd_x_ub, dtype)
    else:
        return pd_x_ub, res_for_gamma

    return pd_x, res_for_gamma


def _get_res(data, params, shape_x, dtype, cast_dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x

    Parameters
    ----------
    data: dict
        placeholders after cast
    params: dict
        {"param_axis": param_axis, "reduce_axis": reduce_axis,
        "mean_num": mean_num}
    shape_x: list or tuple
        shape of x
    dtype: str
        the data type
    cast_dtype: str
        if api_check_support float32, then equal to float32 else float16

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    pd_x, res_for_gamma = _get_pd_x(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


def _get_pds(data_dy, data_x, data_variance, data_mean, data_gamma, shape_gamma_ori):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.

    Parameters
    ----------
    data_dy: TVM tensor
        the placeholder of dy input data
    data_x: TVM tensor
        the placeholder of x input data
    data_variance: TVM tensor
        the placeholder of variance input data
    data_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma_ori: list or tuple
        original shape of gamma

    Returns
    -------
    pd_x: tvm.tensor
        partial derivation of x
    pd_gamma: tvm.tensor
        partial derivation of gamma
    pd_beta: tvm.tensor
        partial derivation of beta
    """
    dtype = data_dy.dtype.lower()
    shape_x = shape_util.shape_to_list(data_x.shape)
    shape_mean = shape_util.shape_to_list(data_mean.shape)

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        cast_dtype = "float32"
    params = _get_params(shape_x, shape_mean, shape_gamma_ori)

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_variance = tbe.cast_to(data_variance, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")
        data_gamma = tbe.cast_to(data_gamma, "float32")

    data = {
        "data_dy"      : data_dy,
        "data_x"       : data_x,
        "data_variance": data_variance,
        "data_mean"    : data_mean,
        "data_gamma"   : data_gamma
    }

    pd_x, res_for_gamma = _get_res(data, params, shape_x, dtype, cast_dtype)

    return pd_x, res_for_gamma


@tbe_platform.fusion_manager.fusion_manager.register("layer_norm_x_backprop_v2")
def layer_norm_x_backprop_v2_compute(input_dy,
                                     input_x,
                                     input_variance,
                                     input_mean,
                                     input_gamma,
                                     output_pd_x,
                                     kernel_name="layer_norm_x_backprop_v2"):
    """
    DSL description of the layernorm_grad operator's mathematical
    calculation process

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_pd_x: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    res_tuple: tuple
        (pd_x, pd_gamma, pd_beta)
    """
    pd_x, res_for_gamma = _get_pds(input_dy, input_x, input_variance, input_mean, input_gamma, input_gamma.shape)
    res_list = [pd_x, res_for_gamma]

    return res_list


def update_shape_nz(shape_x, shape_var, shape_gamma):
    """
    function of updating Nz shape

    """
    # ND shape of x >= two dim
    # Nz shape of x >= four dim
    len_x = len(shape_x)
    nz_begin = len_x - 4
    shape_x_nz = []
    for i in range(0, nz_begin):
        shape_x_nz.append(shape_x[i])
    shape_x_nz.append(shape_x[nz_begin])
    shape_x_nz.append(shape_x[nz_begin + 1] * shape_x[nz_begin + 2])
    shape_x_nz.append(shape_x[nz_begin + 2])

    # ND shape of var >= two dim
    shape_var_nz = []
    len_var = len(shape_var)
    var_nz_begin = len_var - 2
    for i in range(0, var_nz_begin):
        shape_var_nz.append(shape_var[i])
    shape_var_nz.append(1)
    shape_var_nz.append(shape_var[var_nz_begin])
    shape_var_nz.append(1)

    # ND shape of gamma is one dim
    shape_gamma_nz = []
    for i in range(0, nz_begin):
        shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin])
    shape_gamma_nz.append(1)
    shape_gamma_nz.append(shape_x[nz_begin + 2])

    reduce_nz_axis = []
    param_nz_axis = []
    for i, (xtem, var) in enumerate(zip(shape_x_nz, shape_var_nz)):
        if xtem != var:
            reduce_nz_axis.append(i)

    for i, (xtem, gamma) in enumerate(zip(shape_x_nz, shape_gamma_nz)):
        if xtem != gamma or (xtem == 1 and gamma == 1):
            param_nz_axis.append(i)

    mean_nz_num = 1.0
    for i in reduce_nz_axis:
        mean_nz_num *= shape_x_nz[i]

    param_nz = {
        "shape_x_nz"    : shape_x_nz,
        "shape_var_nz"  : shape_var_nz,
        "shape_gamma_nz": shape_gamma_nz,
        "reduce_axis"   : reduce_nz_axis,
        "param_axis"    : param_nz_axis,
        "mean_num"      : mean_nz_num
    }

    return param_nz


def _get_data_nz(param_nz, dtype):
    """
    get placeholders of data_dy, data_x, data_variance, data_mean and data_gamma

    """
    data_dy = tvm.placeholder(param_nz.get("shape_x_nz"), name="data_dy", dtype=dtype)
    data_x = tvm.placeholder(param_nz.get("shape_x_nz"), name="data_x", dtype=dtype)
    data_variance = tvm.placeholder(param_nz.get("shape_var_nz"), name="data_variance", dtype=dtype)
    data_mean = tvm.placeholder(param_nz.get("shape_var_nz"), name="data_mean", dtype=dtype)
    data_gamma = tvm.placeholder(param_nz.get("shape_gamma_nz"), name="data_gamma", dtype=dtype)

    data_gm = (data_dy, data_x, data_variance, data_mean, data_gamma)

    return data_gm


def _get_pd_xl_nz(data, param_nz):
    """
    compute pd_xl according to data_dy, data_gamma and shape_x

    """
    data_gamma_cast = tbe.broadcast(data.get("data_gamma"), param_nz.get("shape_x_nz"))
    pd_xl = tbe.vmul(data_gamma_cast, data.get("data_dy"))

    return pd_xl


def _get_pd_var_front_nz(data, cast_dtype):
    """
    compute front part of pd_var according to data_variance

    """
    var_elta = tbe.vadds(data.get("data_variance"), tvm.const(EPSLON, dtype=cast_dtype))
    var_elta_log = tbe.vlog(var_elta)
    var_elta_mul = tbe.vmuls(var_elta_log, tvm.const(-0.5, dtype=cast_dtype))
    var_elta_2 = tbe.vexp(var_elta_mul)
    pdvar1_mul = tbe.vmul(var_elta_2, var_elta_2)
    pd_var_1 = tbe.vmul(pdvar1_mul, var_elta_2)

    return pd_var_1, var_elta_2


def _get_pd_var_nz(data, param_nz, pd_xl, cast_dtype):
    """
    compute pd_var according to data_x, data_mean, reduce_axis and pd_xl

    """
    pd_var_1, var_elta_2 = _get_pd_var_front_nz(data, cast_dtype)

    data_mean_cast = tbe.broadcast(data.get("data_mean"), param_nz.get("shape_x_nz"))
    sub_x_mean = tbe.vsub(data.get("data_x"), data_mean_cast)

    pdvar_mul1 = tbe.vmul(sub_x_mean, pd_xl)
    pdvar_sum = tbe.sum(pdvar_mul1, param_nz.get("reduce_axis"), keepdims=True)
    pdvar_mul3 = tbe.vmul(pdvar_sum, pd_var_1)
    pd_var = tbe.vmuls(pdvar_mul3, tvm.const(-0.5, dtype=cast_dtype))

    return pd_var, var_elta_2, sub_x_mean


def _get_pd_mean_nz(param_nz, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype):
    """
    compute pd_mean according to reduce_axis, pd_xl, pd_var,
    var_elta_2 and sub_x_mean

    """
    pdmean1_sum = tbe.sum(pd_xl, param_nz.get("reduce_axis"), keepdims=True)
    pdmean1_mul = tbe.vmul(pdmean1_sum, var_elta_2)
    pd_mean_1 = tbe.vmuls(pdmean1_mul, tvm.const(-1.0, dtype=cast_dtype))

    return pd_mean_1


def _get_pd_x_nz(data, param_nz, dtype, cast_dtype):
    """
    compute pd_x according to data, params and shape_x

    """
    pd_xl = _get_pd_xl_nz(data, param_nz)

    pd_var, var_elta_2, sub_x_mean = _get_pd_var_nz(data, param_nz, pd_xl, cast_dtype)

    pd_mean = _get_pd_mean_nz(param_nz, pd_xl, pd_var, var_elta_2, sub_x_mean, cast_dtype)

    var_elta_2_cast = _broadcast_nz(var_elta_2, param_nz.get("shape_x_nz"))
    pd_x_1 = tbe.vmul(var_elta_2_cast, pd_xl)
    res_for_gamma = tbe.vmul(var_elta_2_cast, sub_x_mean)
    pd_var = tbe.vmuls(pd_var, tvm.const((2 * (param_nz.get("mean_num") ** (-1))), dtype=cast_dtype))
    pdx2_broad = _broadcast_nz(pd_var, param_nz.get("shape_x_nz"))
    pd_x_2 = tbe.vmul(pdx2_broad, sub_x_mean)
    pd_x_3 = tbe.vmuls(pd_mean, tvm.const((param_nz.get("mean_num") ** (-1)), dtype=cast_dtype))

    pdx_broad = _broadcast_nz(pd_x_3, param_nz.get("shape_x_nz"))
    pdx_add = tbe.vadd(pd_x_1, pd_x_2)
    pd_x_ub = tbe.vadd(pdx_add, pdx_broad)

    if dtype == "float16" and cast_dtype == "float32":
        pd_x = tbe.cast_to(pd_x_ub, dtype)
    else:
        return pd_x_ub, res_for_gamma

    return pd_x, res_for_gamma


def _get_res_nz(data, param_nz, dtype, cast_dtype):
    """
    compute pd_x, pd_gamma, pd_beta according to data, params and shape_x

    """
    pd_x, res_for_gamma = _get_pd_x_nz(data, param_nz, dtype, cast_dtype)

    return pd_x, res_for_gamma


def _get_pds_nz(data_dy, data_x, data_variance, data_mean, data_gamma, param_nz):
    """
    get params and data, compute pd_x, pd_gamma, pd_beta.

    """
    dtype = data_dy.dtype.lower()

    has_improve_precision = False
    cast_dtype = dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support("te.lang.cce.vexp", "float32"):
        has_improve_precision = True
        cast_dtype = "float32"

    if has_improve_precision:
        data_dy = tbe.cast_to(data_dy, "float32")
        data_x = tbe.cast_to(data_x, "float32")
        data_variance = tbe.cast_to(data_variance, "float32")
        data_mean = tbe.cast_to(data_mean, "float32")
        data_gamma = tbe.cast_to(data_gamma, "float32")

    data = {
        "data_dy"      : data_dy,
        "data_x"       : data_x,
        "data_variance": data_variance,
        "data_mean"    : data_mean,
        "data_gamma"   : data_gamma
    }

    pd_x, res_for_gamma = _get_res_nz(data, param_nz, dtype, cast_dtype)

    return pd_x, res_for_gamma


def layer_norm_x_back_nz_compute(data_dy, data_x, data_variance, data_mean, data_gamma, param_nz):
    """
    DSL description of the layernorm_grad operator's mathematical
    calculation process

    Parameters
    ----------
    data_dy: TVM tensor
        the placeholder of dy input data
    data_x: TVM tensor
        the placeholder of x input data
    data_variance: TVM tensor
        the placeholder of variance input data
    data_mean: TVM tensor
        the placeholder of mean input data
    data_gamma: TVM tensor
        the placeholder of gamma input data
    shape_gamma_ori: list or tuple
        original shape of gamma

    Returns
    -------
    res_tuple: tuple
        (pd_x, res_for_gamma)
    """
    pd_x, res_for_gamma = _get_pds_nz(data_dy, data_x, data_variance, data_mean, data_gamma, param_nz)

    return [pd_x, res_for_gamma]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def layer_norm_x_backprop_v2(input_dy,
                             input_x,
                             input_variance,
                             input_mean,
                             input_gamma,
                             output_pd_x,
                             res_for_gamma,
                             kernel_name="layer_norm_x_backprop_v2"):
    """
    algorithm: layernorm_grad
    calculating: gradient of layernorm
                 compute partial derivation of x, gamma and beta
        pd_xl    = data_dy*data_gamma
        pd_var   = np.sum(((-0.5)*pd_xl*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-1.5))),
                   reduce_axis, keepdims=True)
        pd_mean  = np.sum(((-1.0)*pd_xl
                   *np.power((data_variance + EPSLON), (-0.5))),
                   reduce_axis, keepdims=True)
                   + pd_var*(1.0/m)
                   *np.sum(((-2.0)*(data_x - data_mean)),
                   reduce_axis, keepdims=True)
        pd_x     = pd_xl*np.power((data_variance + EPSLON), (-0.5))
                   + pd_var*(2.0/m)*(data_x - data_mean) + pd_mean*(1.0/m)
        pd_gamma = np.sum((data_dy*(data_x - data_mean)
                   *np.power((data_variance + EPSLON), (-0.5))),
                   param_axis, keepdims=True)
        pd_beta  = np.sum(data_dy, param_axis, keepdims=True)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of input dy, only support float16, float32
    input_x: dict
        shape and dtype of input x, only support float16, float32
    input_variance: dict
        shape and dtype of input variance, only support float16, float32
    input_mean: dict
        shape and dtype of input mean, only support float16, float32
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32
    output_y: dict
        shape and dtype of output, only support float16, float32
    res_for_gamma: dict
        shape and dtype of output, only support float16, float32
    kernel_name: str
        cce kernel name, default value is "layer_norm_x_backprop_v2"

    Returns
    -------
    None
    """
    dtype = input_dy.get("dtype").lower()
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_variance = input_variance.get("shape")
    shape_mean = input_mean.get("shape")
    shape_gamma = input_gamma.get("shape")
    format_dy = input_dy.get("format")
    global EPSLON
    EPSLON = 1e-5 if dtype == "float16" else 1e-12
    if layer_norm_x_backprop_v2_unify.is_special_cases(shape_dy, shape_variance, shape_gamma):
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            context.add_addition("is_static", True)
            layer_norm_x_backprop_v2_unify.layer_norm_x_backprop_v2(input_dy, input_x, input_variance, input_mean,
                                                                    input_gamma, output_pd_x, res_for_gamma,
                                                                    kernel_name)
        else:
            with tbe_context.op_context.OpContext("static"):
                tbe_context.op_context.get_context().add_addition("is_static", True)
                layer_norm_x_backprop_v2_unify.layer_norm_x_backprop_v2(input_dy, input_x, input_variance, input_mean,
                                                                        input_gamma, output_pd_x, res_for_gamma,
                                                                        kernel_name)
        return
    else:
        if format_dy.upper() == "FRACTAL_NZ":
            param_nz = update_shape_nz(shape_x, shape_variance, shape_gamma)
            data_dy = tvm.placeholder(param_nz.get("shape_x_nz"), name="data_dy", dtype=dtype)
            data_x = tvm.placeholder(param_nz.get("shape_x_nz"), name="data_x", dtype=dtype)
            data_variance = tvm.placeholder(param_nz.get("shape_var_nz"), name="data_variance", dtype=dtype)
            data_mean = tvm.placeholder(param_nz.get("shape_var_nz"), name="data_mean", dtype=dtype)
            data_gamma = tvm.placeholder(param_nz.get("shape_gamma_nz"), name="data_gamma", dtype=dtype)

            res_list = layer_norm_x_back_nz_compute(data_dy, data_x, data_variance, data_mean, data_gamma, param_nz)

            tensor_list = [data_dy, data_x, data_variance, data_mean, data_gamma] + res_list

            with tvm.target.cce():
                sch = tbe.auto_schedule(res_list)

            config = {"print_ir": False, "name": kernel_name, "tensor_list": tensor_list}

            tbe.cce_build_code(sch, config)
        else:
            _check_params({
                "shape_dy"   : shape_dy,
                "shape_x"    : shape_x,
                "shape_var"  : shape_variance,
                "shape_mean" : shape_mean,
                "shape_gamma": shape_gamma,
                "dtype"      : dtype,
                "kernel_name": kernel_name
            })

            shape_gamma = _update_gamma_shape(shape_x, shape_gamma)[0]

            data_gm = _get_data_gm(
                {
                    "shape_dy"   : shape_dy,
                    "shape_x"    : shape_x,
                    "shape_var"  : shape_variance,
                    "shape_mean" : shape_mean,
                    "shape_gamma": shape_gamma
                }, dtype)

            res_list = layer_norm_x_backprop_v2_compute(data_gm[0], data_gm[1], data_gm[2], data_gm[3], data_gm[4],
                                                        output_pd_x)
            with tvm.target.cce():
                sch = tbe.auto_schedule(res_list)

            tensor_list = list(data_gm) + list(res_list)

            config = {"print_ir": False, "name": kernel_name, "tensor_list": tensor_list}

            tbe.cce_build_code(sch, config)
