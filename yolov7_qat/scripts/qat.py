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
import sys
import os

# Add the current directory to PYTHONPATH for YoloV7
sys.path.insert(0, os.path.abspath("."))
pydir = os.path.dirname(__file__)

import yaml
import collections
import warnings
import argparse
import json
from pathlib import Path

# PyTorch
import torch

# YoloV7
import test
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.google_utils import attempt_download
from utils.general import init_seeds

import quantization.quantize as quantize

# Disable all warning
warnings.filterwarnings("ignore")


class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


# Load YoloV7 Model
def load_yolov7_model(weight, device) -> Model:

    attempt_download(weight)
    model = torch.load(weight, map_location=device)["model"]
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model


def create_coco_train_dataloader(cocodir, batch_size=10):

    with open("data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    loader = create_dataloader(
        f"{cocodir}/train2017.txt", 
        imgsz=640, 
        batch_size=batch_size, 
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=True, hyp=hyp, rect=False, cache=False, stride=32,pad=0, image_weights=False)[0]
    return loader


def create_coco_val_dataloader(cocodir, batch_size=10, keep_images=None):

    loader = create_dataloader(
        f"{cocodir}/val2017.txt", 
        imgsz=640, 
        batch_size=batch_size, 
        length=keep_images,
        opt=collections.namedtuple("Opt", "single_cls")(False),
        augment=False, hyp=None, rect=True, cache=False,stride=32,pad=0.5, image_weights=False)[0]
    return loader


def evaluate_coco(model, dataloader, using_cocotools = False, save_dir=".", conf_thres=0.001, iou_thres=0.65):

    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        "data/coco.yaml", 
        save_dir=Path(save_dir),
        dataloader=dataloader, conf_thres=conf_thres,iou_thres=iou_thres,model=model,is_coco=True,
        plots=False,half_precision=True,save_json=using_cocotools)[0][3]
    

def export_onnx(model : Model, file, size=640, dynamic_batch=False):

    device = next(model.parameters()).device
    model.float()

    dummy = torch.zeros(1, 3, size, size, device=device)
    model.model[-1].concat = True
    grid_old_func = model.model[-1]._make_grid
    model.model[-1]._make_grid = lambda *args: torch.from_numpy(grid_old_func(*args).data.numpy())

    quantize.export_onnx(model, dummy, file, opset_version=13, 
        input_names=["images"], output_names=["outputs"], 
        dynamic_axes={"images": {0: "batch"}, "outputs": {0: "batch"}} if dynamic_batch else None
    )
    model.model[-1].concat = False
    model.model[-1]._make_grid = grid_old_func


def cmd_quantize(weight, cocodir, device, ignore_policy, save_ptq, save_qat, supervision_stride, iters, eval_origin, eval_ptq):
    quantize.initialize()

    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)
    
    device  = torch.device(device)
    model   = load_yolov7_model(weight, device)
    train_dataloader = create_coco_train_dataloader(cocodir)
    val_dataloader   = create_coco_val_dataloader(cocodir)
    quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)
    quantize.apply_custom_rules_to_quantizer(model, export_onnx)
    quantize.calibrate_model(model, train_dataloader)

    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
            summary.append(["Origin", ap])

    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
        summary.append(["PTQ", ap])

    if save_ptq:
        print(f"Save ptq model to {save_ptq}")
        torch.save({"model": model}, save_ptq)

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    best_ap = 0
    def per_epoch(model, epoch, lr):

        nonlocal best_ap
        ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
        summary.append([f"QAT{epoch}", ap])

        if ap > best_ap:
            print(f"Save qat model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, save_qat)

    def preprocess(datas):
        return datas[0].to(device).float() / 255.0

    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        def impl(name, module):
            if id(module) not in supervision_list: return False
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            else:
                print(f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
            return idx in keep_idx
        return impl

    quantize.finetune(
        model, train_dataloader, per_epoch, early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess, supervision_policy=supervision_policy())


def cmd_export(weight, save, size, dynamic):
    
    quantize.initialize()
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
        
    export_onnx(torch.load(weight, map_location="cpu")["model"], save, size, dynamic_batch=dynamic)
    print(f"Save onnx to {save}")


def cmd_sensitive_analysis(weight, device, cocodir, summary_save, num_image):

    quantize.initialize()
    device  = torch.device(device)
    model   = load_yolov7_model(weight, device)
    train_dataloader = create_coco_train_dataloader(cocodir)
    val_dataloader   = create_coco_val_dataloader(cocodir, keep_images=None if num_image is None or num_image < 1 else num_image)
    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_coco(model, val_dataloader)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quantize.disable_quantization(layer).apply()
            ap = evaluate_coco(model, val_dataloader)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")
    
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


def cmd_test(weight, device, cocodir, confidence, nmsthres):

    device  = torch.device(device)
    model   = load_yolov7_model(weight, device)
    val_dataloader   = create_coco_val_dataloader(cocodir)
    evaluate_coco(model, val_dataloader, True, conf_thres=confidence, iou_thres=nmsthres)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='qat.py')
    subps  = parser.add_subparsers(dest="cmd")
    exp    = subps.add_parser("export", help="Export weight to onnx file")
    exp.add_argument("weight", type=str, default="yolov7.pt", help="export pt file")
    exp.add_argument("--save", type=str, required=False, help="export onnx file")
    exp.add_argument("--size", type=int, default=640, help="export input size")
    exp.add_argument("--dynamic", action="store_true", help="export dynamic batch")

    qat = subps.add_parser("quantize", help="PTQ/QAT finetune ...")
    qat.add_argument("weight", type=str, nargs="?", default="yolov7.pt", help="weight file")
    qat.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    qat.add_argument("--device", type=str, default="cuda:0", help="device")
    qat.add_argument("--ignore-policy", type=str, default="model\.105\.m\.(.*)", help="regx")
    qat.add_argument("--ptq", type=str, default="ptq.pt", help="file")
    qat.add_argument("--qat", type=str, default=None, help="file")
    qat.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    qat.add_argument("--iters", type=int, default=200, help="iters per epoch")
    qat.add_argument("--eval-origin", action="store_true", help="do eval for origin model")
    qat.add_argument("--eval-ptq", action="store_true", help="do eval for ptq model")

    sensitive = subps.add_parser("sensitive", help="Sensitive layer analysis")
    sensitive.add_argument("weight", type=str, nargs="?", default="yolov7.pt", help="weight file")
    sensitive.add_argument("--device", type=str, default="cuda:0", help="device")
    sensitive.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    sensitive.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")
    sensitive.add_argument("--num-image", type=int, default=None, help="number of image to evaluate")

    testcmd = subps.add_parser("test", help="Do evaluate")
    testcmd.add_argument("weight", type=str, default="yolov7.pt", help="weight file")
    testcmd.add_argument("--cocodir", type=str, default="/datav/dataset/coco", help="coco directory")
    testcmd.add_argument("--device", type=str, default="cuda:0", help="device")
    testcmd.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    testcmd.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")

    args = parser.parse_args()
    init_seeds(57)

    if args.cmd == "export":
        cmd_export(args.weight, args.save, args.size, args.dynamic)
    elif args.cmd == "quantize":
        print(args)
        cmd_quantize(
            args.weight, args.cocodir, args.device, args.ignore_policy, 
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq
        )
    elif args.cmd == "sensitive":
        cmd_sensitive_analysis(args.weight, args.device, args.cocodir, args.summary, args.num_image)
    elif args.cmd == "test":
        cmd_test(args.weight, args.device, args.cocodir, args.confidence, args.nmsthres)
    else:
        parser.print_help()
