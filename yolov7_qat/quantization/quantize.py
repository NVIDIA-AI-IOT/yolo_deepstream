################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

# PyTorch
import torch
import torch.optim as optim
from torch.cuda import amp

# Pytorch Quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from absl import logging as quant_logging

# Custom Rules
from quantization.rules import find_quantizer_pairs


class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


# Initialize PyTorch Quantization
def initialize():
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_logging.set_verbosity(quant_logging.ERROR)


def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):

    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy : Union[str, List[str], Callable], path : str) -> bool:

    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False


def replace_to_quantization_module(model : torch.nn.Module, ignore_policy : Union[str, List[str], Callable] = None):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    print(f"Quantization: {path} has ignored.")
                    continue
                    
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)

        if len(names) == 1:
            return value

        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))


def apply_custom_rules_to_quantizer(model : torch.nn.Module, export_onnx : Callable):

    # apply rules to graph
    export_onnx(model, "quantization-custom-rules-temp.onnx")
    pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
    for major, sub in pairs:
        print(f"Rules: {sub} match to {major}")
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
    os.remove("quantization-custom-rules-temp.onnx")


def calibrate_model(model : torch.nn.Module, dataloader, num_batch=25):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
        
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                imgs = datas[0].to(device, non_blocking=True).float() / 255.0
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    device = next(model.parameters()).device
    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")


def finetune(
    model : torch.nn.Module, train_dataloader, per_epoch_callback : Callable = None, preprocess : Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule : Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy : Callable = None
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()

    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers

    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device       = next(model.parameters()).device

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-6
        }

    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])


    for iepoch in range(nepochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs  = []
        origin_outputs = []
        remove_handle  = []

        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):

            if ibatch >= early_exit_batchs_per_epoch:
                break
            
            if preprocess:
                imgs = preprocess(imgs)
                
            imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for index, (mo, fo) in enumerate(zip(model_outputs, origin_outputs)):
                    quant_loss += quant_lossfn(mo, fo)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break


def export_onnx(model, input, file, *args, **kwargs):

    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False
