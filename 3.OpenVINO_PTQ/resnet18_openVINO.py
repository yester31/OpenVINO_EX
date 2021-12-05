import torch, torchvision, os, cv2, struct, time
import numpy as np
from utils import *
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
        net = torch.load('model/resnet18.pth')              # resnet18.pth 파일 로드
    else:                                                   # resnet18.pth 파일이 없다면
        net = torchvision.models.resnet18(pretrained=True)  # torchvision에서 resnet18 pretrained weight 다운로드 수행
        torch.save(net, 'model/resnet18.pth')               # resnet18.pth 파일 저장

    batch_size = 1
    net.eval()
    #half = True
    half = False
    if half:
        net.half()  # to FP16

    # Convert the Pytorch model to ONNX ================================================================================
    dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    # torch_out = net(dummy_input)
    onnx_model_name = "model/resnet18_cpu.onnx"
    if not os.path.isfile(onnx_model_name):
        with torch.no_grad():
            # 모델 변환
            torch.onnx.export(net,                  # 실행될 모델
                              dummy_input,          # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                              onnx_model_name,      # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                              export_params=True,   # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                              opset_version=11,     # 모델을 변환할 때 사용할 ONNX 버전
                              do_constant_folding=True, # 최적하시 상수폴딩을 사용할지의 여부
                              input_names =['input'],   # 모델의 입력값을 가리키는 이름
                              output_names=['output'],  # 모델의 출력값을 가리키는 이름
                              dynamic_axes={'input': {0: 'batch_size'},  # 가변적인 길이를 가진 차원
                                            'output': {0: 'batch_size'}})
            print('Model has been converted to ONNX')
    # Convert the ONNX model to OpenVINO ================================================================================
    # Construct the command for Model Optimizer
    mo_command = f"""mo
                     --input_model "{onnx_model_name}"
                     --input_shape "[1,3, {224}, {224}]"
                     --data_type FP16
                     --output_dir "model"
                     """
    mo_command = " ".join(mo_command.split())
    print("Model Optimizer command to convert the ONNX model to OpenVINO:")

    ir_path = onnx_model_name.split('.')[0] + (".xml")
    if not os.path.isfile(ir_path):
        print("Exporting ONNX model to IR... This may take a few minutes.")
        print(mo_command)
        os.system(mo_command)
    else:
        print(f"IR model {ir_path} already exists.")

    ir_path_ = onnx_model_name.split('.')[0]
    net_onnx = ie.read_network(model=ir_path_ +'.xml')
    exec_net_onnx = ie.load_network(network=net_onnx, device_name=intel_device)

    input_layer_onnx = next(iter(exec_net_onnx.input_info))
    output_layer_onnx = next(iter(exec_net_onnx.outputs))

    img = cv2.imread('date/panda0.jpg')  # image file load
    dur_time = 0
    iteration = 100

    # 속도 측정에서 첫 1회 연산 제외하기 위한 계산
    input_ = preprocess_(img, half)
    res_onnx = exec_net_onnx.infer(inputs={input_layer_onnx: input_})
    res_onnx = res_onnx[output_layer_onnx]

    for i in range(iteration):
        begin = time.time()
        input_ = preprocess_(img, half)
        res_onnx = exec_net_onnx.infer(inputs={input_layer_onnx: input_})
        res_onnx = res_onnx[output_layer_onnx]
        dur = time.time() - begin
        dur_time += dur
        #print('{} dur time : {}'.format(i, dur))

    print('{} iteration time : {} [sec]'.format(iteration, dur_time))

    max_index = np.argmax(res_onnx[0])
    max_value = res_onnx[0, max_index]
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
# 100 iteration time : 0.865093469619751 [sec]
# device = "gpu" 일 때 (Intel(R) Iris(R) Xe Graphics (iGPU))
# 100 iteration time : 0.35637593269348145 [sec]