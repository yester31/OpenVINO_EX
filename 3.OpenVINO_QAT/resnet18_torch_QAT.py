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
import copy, random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
print(f"Using {device} device")

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():

    set_random_seeds(random_seed=0)
    half = False
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

    batch_size = 128
    image_size = 224

    # Data loading code
    val_dir = "F:\dataset\imagenet_dataset\ILSVRC2012_img_val"
    train_dir = "F:\dataset\imagenet_dataset\ILSVRC2012_img_train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_dir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)
    # acc1 = validate(val_loader, model, criterion, device)
    # print(f"Accuracy of FP32 model: {acc1:.3f}") #  * Acc@1 69.756 Acc@5 89.084
    print('fused start!')
    model.to('cpu:0')
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
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                #torch.quantization.fuse_modules(basic_block, [["conv1", "bn1"], ["conv2", "bn2"]], inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    # Model and fused model should be equivalent.
    model.eval()
    fused_model.eval()
    assert model_equivalence(model_1=model,model_2=fused_model,device=device,rtol=1e-03,atol=1e-06, num_tests=1,input_size=(1, 3, 224, 224)), "Fused model is not equivalent to the original model!"
    print('fused model done!')

    # Prepare the model for quantization aware training. This inserts observers in
    # the model that will observe activation tensors during calibration.
    quantized_model = QuantizedResNet18(model_fp32=fused_model).train()
    # Select quantization schemes from
    # https://pytorch.org/docs/stable/quantization-support.html
    # quantized_model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    quantized_model.qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))

    # Print quantization configurations
    print(quantized_model.qconfig)

    # https://pytorch.org/docs/stable/_modules/torch/quantization/quantize.html#prepare_qat
    torch.quantization.prepare_qat(quantized_model, inplace=True)

    print("Training QAT Model...")
    quantized_model.train()
    train_model(model=quantized_model,
                train_loader=train_loader,
                test_loader=val_loader,
                device=device,
                learning_rate=1e-3,
                num_epochs=1)
    quantized_model.to('cpu:0')

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model.eval()
    # Print quantized model.
    # print(quantized_model)
    quantized_model.to('cpu:0')
    model_filepath = "/model/resnet18_int8_jit.pth"

    # Save quantized model.
    with torch.no_grad():
        torch.jit.save(torch.jit.script(quantized_model), model_filepath)

    # Load quantized model.
    quantized_jit_model = torch.jit.load(model_filepath, map_location=device)

    img = cv2.imread('date/panda0.jpg')  # image file load
    dur_time = 0
    iteration = 100

    # 속도 측정에서 첫 1회 연산 제외하기 위한 계산
    out = infer(img, quantized_jit_model, half, device)
    torch.cuda.synchronize()

    for i in range(iteration):
        begin = time.time()
        out = infer(img, quantized_jit_model, half, device)
        torch.cuda.synchronize()
        dur = time.time() - begin
        dur_time += dur
        #print('{} dur time : {}'.format(i, dur))

    print('{} iteration time : {} [sec]'.format(iteration, dur_time))

    max_tensor = out.max(dim=1)
    max_value = max_tensor[0].cpu().data.numpy()[0]
    max_index = max_tensor[1].cpu().data.numpy()[0]
    print('resnet18 max index : {} , value : {}, class name : {}'.format(max_index, max_value, class_name[max_index] ))




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