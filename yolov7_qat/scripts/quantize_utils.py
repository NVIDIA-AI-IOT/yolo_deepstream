################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Variable
import onnx
import numpy as np
import argparse
import logging

LAYER_ID = 0
TENSOR_ID = 0
def get_qparams_constants(node_to_quantize_name, scale_init=0.5, zero_point_init=0):
    global LAYER_ID, TENSOR_ID
    """ ATTENTION: "node_to_quantize_name" needs to be different every time this function is called.
    Otherwise, "scale, zero_point" are overwritten.
    TODO: ensure that this happens! The same goes for
        "q_out and dq_out = gs.Variable(UNIQUE_NAME)"

    :param node_to_quantize_name:
    :param scale_init:
    :param zero_point_init:
    :return: 2 gs.Constants (scale and zero-point).
    """
    scale = gs.Constant(
        name=node_to_quantize_name + "_scale" + str(TENSOR_ID),
        values=np.array(scale_init, dtype=np.float32))
    TENSOR_ID = TENSOR_ID + 1
    zero_point = gs.Constant(
        name=node_to_quantize_name + "_zero_point" + str(TENSOR_ID),
        values=np.array(zero_point_init, dtype=np.int8)
    )
    TENSOR_ID = TENSOR_ID + 1
    return scale, zero_point

def quantize_tensor(graph, tensor_to_quantize, scale, name_suffix=""):
    global LAYER_ID, TENSOR_ID
    output_nodes = tensor_to_quantize['x'].outputs
    nodes_and_quantized = []
    nodes_inputidx = []

    for node in output_nodes:
        for idx, inp in enumerate(node.inputs):
            if inp.name == tensor_to_quantize['x'].name:
                nodes_and_quantized.append(node)
                nodes_inputidx.append(idx)
                break

    # QuantizeLinear node
    q_scale, q_zero_point = get_qparams_constants(tensor_to_quantize['x'].name + "_inp_q" + name_suffix, scale_init=scale)
    q_out = gs.Variable(name=tensor_to_quantize['x'].name + "_QuantizeLinear_out" + name_suffix + str(TENSOR_ID))
    TENSOR_ID = TENSOR_ID + 1
    quant_node = gs.Node(
        op="QuantizeLinear",
        name="QuantI_"+ tensor_to_quantize['x'].name + str(LAYER_ID),
        inputs=[tensor_to_quantize["x"], q_scale, q_zero_point],
        outputs=[q_out]
    )
    LAYER_ID = LAYER_ID + 1
    # DequantizeLinear node
    dq_scale, dq_zero_point = get_qparams_constants(tensor_to_quantize['x'].name + "_inp_dq" + name_suffix, scale_init=scale)
    dq_out = gs.Variable(name=tensor_to_quantize['x'].name + "_DequantizeLinear_out" + name_suffix  + str(TENSOR_ID))
    TENSOR_ID = TENSOR_ID + 1
    dequant_node = gs.Node(
        op="DequantizeLinear",
        name="DequantI_"+ tensor_to_quantize['x'].name + str(LAYER_ID),
        inputs=[q_out, dq_scale, dq_zero_point],
        outputs=[dq_out]
    )
    LAYER_ID = LAYER_ID + 1
    #shit code
    for i, node in enumerate(nodes_and_quantized):
        node.inputs[nodes_inputidx[i]] = dq_out

    graph.nodes.extend([quant_node, dequant_node])
    return graph

def quantize_input(graph, node_to_quantize, node_to_quantize_input, scale, name_suffix=""):
    global LAYER_ID, TENSOR_ID
    # QuantizeLinear node
    q_scale, q_zero_point = get_qparams_constants(node_to_quantize.name + "_inp_q" + name_suffix, scale_init=scale)
    q_out = gs.Variable(name=node_to_quantize.name + "_QuantizeLinear_out" + name_suffix  + name_suffix + str(TENSOR_ID))
    TENSOR_ID = TENSOR_ID + 1
    quant_node = gs.Node(
        op="QuantizeLinear",
        name="QuantI_"+ node_to_quantize.name + str(LAYER_ID),
        inputs=[node_to_quantize_input["x"], q_scale, q_zero_point],
        outputs=[q_out]
    )
    LAYER_ID = LAYER_ID + 1

    # DequantizeLinear node
    dq_scale, dq_zero_point = get_qparams_constants(node_to_quantize.name + "_inp_dq" + name_suffix, scale_init=scale)
    dq_out = gs.Variable(name=node_to_quantize.name + "_DequantizeLinear_out" + name_suffix + name_suffix + str(TENSOR_ID))
    TENSOR_ID = TENSOR_ID + 1
    dequant_node = gs.Node(
        op="DequantizeLinear",
        name="DequantI_"+ node_to_quantize.name + str(LAYER_ID),
        inputs=[q_out, dq_scale, dq_zero_point],
        outputs=[dq_out]
    )
    LAYER_ID = LAYER_ID + 1

    node_to_quantize.inputs[node_to_quantize_input["idx"]] = dq_out
    graph.nodes.extend([quant_node, dequant_node])

    graph.cleanup().toposort()
    return graph


def quantize_weight(graph, node_to_quantize, node_to_quantize_weight, axis=0, name_suffix=""):
    global LAYER_ID, TENSOR_ID
    """
    When connected to the weight, the "y_scale" parameter can be recovered directly from the Weight matrix.
    See official doc: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#intro-quantization

    :param graph:
    :param node_to_quantize:
    :param node_to_quantize_weight:
    :param axis:
    :param name_suffix:
    :return:
    """
    shape = node_to_quantize_weight["x"].shape[axis]
    # Recover "y_scale" from weight matrix
    weight_matrix = node_to_quantize_weight["x"].values
    y_scale_arr = []
    # Recover "y_scale" for each batch. If axis != 0, move the desired axis to the be idx=0.
    if axis !=0:
        weight_matrix = np.moveaxis(weight_matrix, [axis], [0])
    # for bais 1d-weight
    if len(weight_matrix.shape) == 1:
        weight_matrix = np.expand_dims(weight_matrix, axis=0)
    for w in weight_matrix[:]:
        dyn_range = max(abs(w.min()), abs(w.max()))
        y_scale = dyn_range / 127.0
        y_scale_arr.append(y_scale)

    # QuantizeLinear node
    q_scale, q_zero_point = get_qparams_constants(
        node_to_quantize.name + "_weight_q" + name_suffix,
        scale_init=y_scale_arr,  # * np.ones(shape=(shape,)),
        zero_point_init=np.zeros(shape=(shape,)),
    )
    q_out = gs.Variable(name=node_to_quantize.name + "_QuantizeLinear_weight_out" + name_suffix + str(TENSOR_ID))
    TENSOR_ID = TENSOR_ID + 1
    quant_node = gs.Node(
        op="QuantizeLinear",
        name="QuantW_"+ node_to_quantize.name + str(LAYER_ID),
        inputs=[node_to_quantize_weight["x"], q_scale, q_zero_point],
        outputs=[q_out],
        attrs={"axis": axis}
    )
    LAYER_ID = LAYER_ID + 1


    # DequantizeLinear node
    dq_scale, dq_zero_point = get_qparams_constants(
        node_to_quantize.name + "_weight_dq" + name_suffix,
        scale_init=y_scale_arr,  # * np.ones(shape=(shape,)),
        zero_point_init=np.zeros(shape=(shape,)),
    )
    TENSOR_ID = TENSOR_ID + 1
    dq_out = gs.Variable(name=node_to_quantize.name + "_DequantizeLinear_weight_out" + name_suffix + str(TENSOR_ID))
    dequant_node = gs.Node(
        op="DequantizeLinear",
        name="DequantW_"+ node_to_quantize.name + str(LAYER_ID),
        inputs=[q_out, dq_scale, dq_zero_point],
        outputs=[dq_out],
        attrs={"axis": axis}
    )
    LAYER_ID = LAYER_ID + 1

    node_to_quantize.inputs[node_to_quantize_weight["idx"]] = dq_out
    graph.nodes.extend([quant_node, dequant_node])

    graph.cleanup().toposort()
    return graph

def get_node_to_quantize_infos(node_to_quantize, disableResAdd:bool):
    # Separate inputs into activation ('Variable' type) and weight ('Constant' type).
    node_to_quantize_input = []
    node_to_quantize_weight = []
    for idx, inp in enumerate(node_to_quantize.inputs):
        if isinstance(inp, Variable):
            node_to_quantize_input.append({"x": inp, "idx": idx})
            # residual add, will not work with bias add
            if node_to_quantize.op == "Add" and (not disableResAdd) and len(node_to_quantize_input) == 2:
                node_to_quantize_input = [node_to_quantize_input[0]]
        else:  # Constant
            if (
                    len(node_to_quantize_weight) == 0
                    and node_to_quantize.op not in ["Add", "BatchNormalization"]
                    and len(inp.shape) > 1
            ):
                # 1) Only quantize the Weight, not Bias
                # 2) Do not quantize bias matrix in BiasAdd ops
                # 3) Only save weight matrices with shape > 1 (Conv 4D, MatMul 2D)
                node_to_quantize_weight.append({"x": inp, "idx": idx})

            # for bias add after matmul
            elif(
                    len(node_to_quantize_weight) == 0
                    and node_to_quantize.op =="Add"
                    and isinstance(node_to_quantize.inputs[0], gs.Constant)):
                node_to_quantize_weight.append({"x": inp, "idx": idx})


    return node_to_quantize_input, node_to_quantize_weight

def quantize_node_automatically(graph, node_to_quantize, scale, disableResAdd:bool):
    """
    Quantizes a node according to information in graph.json (generated from the PTQ engine building step.

    :return:
    """
    node_to_quantize_input, node_to_quantize_weight = get_node_to_quantize_infos(node_to_quantize, disableResAdd)

    # Quantize inputs
    input_was_quantized = False
    # Quantizable layer
    for i, node_inp in enumerate(node_to_quantize_input):
        graph = quantize_input(graph, node_to_quantize, node_inp, scale, name_suffix=str(i))
        input_was_quantized = True

    # Quantize weights
    for i, node_weight in enumerate(node_to_quantize_weight):
        if input_was_quantized:
            graph = quantize_weight(
                graph,
                node_to_quantize,
                node_weight,
                axis=1 if node_to_quantize.op in ["MatMul", "ConvTranspose"] else 0,  # TODO: Automatize axis detection. Automatize this by checking the expected layer output and extract axis that matches desired dimension.
                name_suffix=str(i)
            )
    return graph

def quantize_tensor_automatically(graph, tensor_to_quantize, scale):
    """
    Quantizes a tensor

    :return:
    """
    tensor_to_quantize = [{'x':tensor_to_quantize},]
    # Quantizable tensor
    for i, tensor_inp in enumerate(tensor_to_quantize):
        graph = quantize_tensor(graph, tensor_inp, scale, name_suffix=str(i))
    return graph


def quant_one_node(graph, node_name, scale=0.04370, disableResAdd:bool = False):
    nodes = graph.nodes
    node_to_quantize = [x for x in nodes if x.name == node_name]
    if len(node_to_quantize) == 0:
        logging.warning(f'node: ',node_name, "did not found, skip")
    if len(node_to_quantize) > 1:
        logging.error(f'found multiple node named: ',node_name)
    node_to_quantize = node_to_quantize[0]
    graph = quantize_node_automatically(graph, node_to_quantize, scale, disableResAdd)
    return graph

def quant_one_tensor(graph, tensor_name, scale=0.04370):
    # nodes = graph.nodes
    tensors = graph.tensors()
    tensor_to_quantize = [tensor for name, tensor in tensors.items() if tensor.name == tensor_name]
    if len(tensor_to_quantize) == 0:
        logging.warning(f'tensor: ',tensor_name, "did not found, skip")
    if len(tensor_to_quantize) > 1:
        logging.error(f'found multiple tensor named: ',tensor_name)

    tensor_to_quantize = tensor_to_quantize[0]
    graph = quantize_tensor_automatically(graph, tensor_to_quantize, scale)
    return graph

def quant_node_of_list(graph, op_name_list:list, disableResAdd:bool):
    for op in op_name_list:
        graph = quant_one_node(graph, op, disableResAdd=disableResAdd)
        ##TODO: if one element is Conv1:0.03, it should be support
    return graph

def quant_tensor_of_list(graph, tensor_name_list:list):
    for tensor in tensor_name_list:
        graph = quant_one_tensor(graph, tensor)
    return graph

# def quant_all_nodes_of_type():
    # return None

def quant_onnx(model_path, output_model_path, nodes_name_to_quant, tensors_name_to_quant, disableResAdd:bool):
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)
    graph = quant_node_of_list(graph, nodes_name_to_quant, disableResAdd)
    graph = quant_tensor_of_list(graph, tensors_name_to_quant)
    graph.cleanup()
    new_model = gs.export_onnx(graph)
    onnx.save(new_model, output_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='iso_the onnx model with new input and output')
    parser.add_argument('--model', default='model.onnx', type=str, help='the onnx model')
    parser.add_argument('--output_model', default='', type=str, help='the output model')
    parser.add_argument('--nodes', nargs='+', type=str, help='the input nodes list you want to quant',default=[])
    parser.add_argument('--disableResAdd', action='store_true', help='if enabled this flag, residual add will have two inputs')

    parser.add_argument('--tensors', nargs='+', type=str, help='the tensors list you want to quant',default=[])

    args = parser.parse_args()
    print(args)
    quant_onnx(args.model,  args.output_model, args.nodes, args.tensors, args.disableResAdd)
