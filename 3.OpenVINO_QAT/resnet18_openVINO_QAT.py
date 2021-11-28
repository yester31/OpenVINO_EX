import sys

if sys.platform == "win32":
    import distutils.command.build_ext
    import os
    from pathlib import Path

    VS_INSTALL_DIR = r"C:/Program Files (x86)/Microsoft Visual Studio"
    cl_paths = sorted(list(Path(VS_INSTALL_DIR).glob("**/Hostx86/x64/cl.exe")))
    if len(cl_paths) == 0:
        raise ValueError(
            "Cannot find Visual Studio. This notebook requires a C++ compiler. If you installed "
            "a C++ compiler, please add the directory that contains cl.exe to `os.environ['PATH']`."
        )
    else:
        # If multiple versions of MSVC are installed, get the most recent version
        cl_path = cl_paths[-1]
        vs_dir = str(cl_path.parent)
        os.environ["PATH"] += f"{os.pathsep}{vs_dir}"
        # Code for finding the library dirs from
        # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
        d = distutils.core.Distribution()
        b = distutils.command.build_ext.build_ext(d)
        b.finalize_options()
        os.environ["LIB"] = os.pathsep.join(b.library_dirs)
        print(f"Added {vs_dir} to PATH")

import time
import warnings  # to disable warnings on export to ONNX
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
import nncf  # Important - should be imported directly after torch

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
#from openvino.inference_engine import IECore
#from torch.jit import TracerWarning

import torchvision, os, cv2, struct, time
import numpy as np
from utils import *
from pathlib import Path

import torch.onnx
import onnx
from openvino.inference_engine import IECore
print("onnx:", onnx.__version__)
ie = IECore()
for device in ie.available_devices:
    device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

intel_device = 'GPU'
#intel_device = 'CPU'
print(f"intel device : {intel_device}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
print(f"Using {device} device")

def main():

    if not os.path.exists('/model'):  # 저장할 폴더가 없다면
        os.makedirs('/model')  # 폴더 생성
        print('make directory {} is done'.format('/model'))

    if os.path.isfile('model/resnet18.pth'):                # resnet18.pth 파일이 있다면
        model = torch.load('model/resnet18.pth')              # resnet18.pth 파일 로드
    else:                                                   # resnet18.pth 파일이 없다면
        model = torchvision.models.resnet18(pretrained=True)  # torchvision에서 resnet18 pretrained weight 다운로드 수행
        torch.save(model, 'model/resnet18.pth')               # resnet18.pth 파일 저장

    model.to(device)

    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    BASE_MODEL_NAME = "resnet18"
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # Paths where PyTorch, ONNX and OpenVINO IR models will be stored
    fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
    fp32_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".onnx")
    fp32_ir_path = fp32_onnx_path.with_suffix(".xml")
    int8_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".onnx")
    int8_ir_path = int8_onnx_path.with_suffix(".xml")

    batch_size = 256
    image_size = 224

    # Data loading code
    #train_dir = "data/train"
    val_dir = "F:\dataset\imagenet_dataset\ILSVRC2012_img_val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(train_dir, transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=batch_size, shuffle=True,
    #     num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)
    # acc1 = validate(val_loader, model, criterion, device)
    # print(f"Accuracy of FP32 model: {acc1:.3f}") #  * Acc@1 69.756 Acc@5 89.084

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    torch.onnx.export(model, dummy_input, fp32_onnx_path)
    print(f"FP32 ONNX model was exported to {fp32_onnx_path}.")

    # Create and Initialize Quantization (Compressed Model)========================================
    nncf_config_dict = {
        "input_info": {"sample_size": [1, 3, image_size, image_size]},
        "log_dir": str('.'),  # log directory for NNCF-specific logging outputs
        "compression": {
            "algorithm": "quantization",  # specify the algorithm here
        },
    }
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)
    # nncf_config = register_default_init_args(nncf_config, train_loader)
    # compression_ctrl, model = create_compressed_model(model, nncf_config)
    # f32_acc1 = validate(val_loader, model, criterion,device)
    # print(f"Accuracy of initialized INT8 model: {f32_acc1:.3f}")

    # # Fine-tune the Compressed Model================================================================
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)
    # # train for one epoch with NNCF
    # train(train_loader, model, criterion, optimizer, 0, device) # 0 mean 1 epoch
    # # evaluate on validation set after Quantization-Aware Training (QAT case)
    # int8_acc1 = validate(val_loader, model, criterion, device)
    # print(f"Accuracy of tuned INT8 model: {int8_acc1:.3f}")
    #
    # if not int8_onnx_path.exists():
    #     warnings.filterwarnings("ignore", category=TracerWarning)
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     # Export INT8 model to ONNX that is supported by the OpenVINO™ toolkit
    #     compression_ctrl.export_model(int8_onnx_path)
    #     print(f"INT8 ONNX model exported to {int8_onnx_path}.")





if __name__ == '__main__':
    main()

# base model(f32)
# device = "cpu:0" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 2.7593486309051514 [sec]
# device = "gpu:0" 일 때 (NVIDIA GeForce RTX 3060 Laptop GPU)
# 100 iteration time : 0.42092013359069824 [sec]

# jit model(f32)
# device = "cpu:0" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 2.768479824066162 [sec]
# device = "gpu:0" 일 때 (NVIDIA GeForce RTX 3060 Laptop GPU)
# 100 iteration time : 0.36458611488342285 [sec]

# onnx(f32)
# device = "cpu" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 1.2189843654632568 [sec]
# device = "gpu" 일 때 (NVIDIA GeForce RTX 3060 Laptop GPU)
# 100 iteration time : 0.3168525695800781 [sec]

# openVINO(f32)
# device = "cpu" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 1.1100926399230957 [sec]
# device = "gpu" 일 때 (Intel(R) Iris(R) Xe Graphics (iGPU))
# 100 iteration time : 0.5102159976959229 [sec]