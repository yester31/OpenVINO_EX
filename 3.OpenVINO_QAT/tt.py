import torch, torchvision, os, cv2, struct, time
import numpy as np
from utils import *
import torch.onnx
import onnx
from openvino.inference_engine import IECore
import torchvision.transforms as transforms
import torchvision.datasets as datasets
print("onnx:", onnx.__version__)

ie = IECore()
for device in ie.available_devices:
    device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

intel_device = 'GPU'
#intel_device = 'CPU'
print(f"device : {intel_device}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
print(f"device : {device}")

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
    num_classes = 1000  # for ImageNet
    init_lr = 1e-4
    batch_size = 128
    epochs = 4
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
        num_workers=4, pin_memory=True)

    import torch.nn as nn

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    acc1 = validate(val_loader, model, criterion,device)
    print(f"Accuracy of FP32 model: {acc1:.3f}")









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