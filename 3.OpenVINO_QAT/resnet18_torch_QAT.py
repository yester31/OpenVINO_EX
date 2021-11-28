import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision, os, cv2, struct, time
import numpy as np
from utils import *
from pathlib import Path

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

    # Paths where PyTorch, ONNX and OpenVINO IR models will be stored
    fp32_pth_path = Path("model/resnet18_fp32").with_suffix(".pth")
    int8_path = Path("model/resnet18_int8").with_suffix(".pth")

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
    import copy
    # Make a copy of the model for layer fusion
    fused_model = copy.deepcopy(model)
    model.train()
    # The model has to be switched to training mode before any layer fusion.
    # Otherwise the quantization aware training will not work correctly.
    fused_model.train()
    # Fuse the model in place rather manually.
    fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # Model and fused model should be equivalent.
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model,model_2=fused_model,device=device,rtol=1e-03,atol=1e-06, num_tests=100,input_size=(1, 3, 224, 224)), "Fused model is not equivalent to the original model!"

    print('done!')




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