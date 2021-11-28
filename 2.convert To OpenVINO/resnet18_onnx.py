import torch, torchvision, os, cv2, struct, time
import numpy as np
from utils import *
import torch.onnx
import onnx
import onnxruntime
import psutil

print('gpu device count : ', torch.cuda.device_count())
print('device_name : ', torch.cuda.get_device_name(0))
print('gpu available : ', torch.cuda.is_available())
print("onnxruntime:", onnxruntime.__version__)
print("onnx:", onnx.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
#device = torch.device("cpu:0")
print(f"{device}")

def main():
    #half = True
    half = False

    if os.path.isfile('model/resnet18.pth'):               # resnet18.pth 파일이 있다면
        net = torch.load('model/resnet18.pth')             # resnet18.pth 파일 로드
    else:                                                  # resnet18.pth 파일이 없다면
        net = torchvision.models.resnet18(pretrained=True) # torchvision에서 resnet18 pretrained weight 다운로드 수행
        torch.save(net, 'model/resnet18.pth')              # resnet18.pth 파일 저장

    batch_size = 1  # 임의의 수
    net.eval()
    net.to(device)
    if half:
        net.half()  # to FP16

    # 모델에 대한 입력값
    dummy_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to(device)
    torch_out = net(dummy_input)

    onnx_model_name = "model/resnet18_{}.onnx".format(device.type)

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

    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)

    if device.type =='cuda' :
        sess_options = onnxruntime.SessionOptions()
        # # Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
        # # Note that this will increase session creation time so enable it for debugging only.
        sess_options.optimized_model_filepath = "model/optimized_resnet18_{}.onnx".format(device.type)
        # # Please change the value according to best setting in Performance Test Tool result.
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        ort_session = onnxruntime.InferenceSession(onnx_model_name, sess_options)
    else :
        sess_options = onnxruntime.SessionOptions()
        # Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
        # Note that this will increase session creation time, so it is for debugging only.
        sess_options.optimized_model_filepath = "model/optimized_resnet18_{}.onnx".format(device.type)
        # For OnnxRuntime 1.7.0 or later, you can set intra_op_num_threads to set thread number like
        #    sess_options.intra_op_num_threads=4
        # Here we use the default value which is a good choice in most cases.
        # Specify providers when you use onnxruntime-gpu for CPU inference.
        ort_session = onnxruntime.InferenceSession(onnx_model_name, sess_options, providers=['CPUExecutionProvider'])

    img = cv2.imread('date/panda0.jpg')  # image file load
    dur_time = 0
    iteration = 100

    # 속도 측정에서 첫 1회 연산 제외하기 위한 계산
    out = infer_onnx(img, ort_session, half, device)

    for i in range(iteration):
        begin = time.time()
        out = infer_onnx(img, ort_session, half, device)
        dur = time.time() - begin
        dur_time += dur
        #print('{} dur time : {}'.format(i, dur))

    print('{} iteration time : {} [sec]'.format(iteration, dur_time))

    max_index = np.argmax(out)
    max_value = out[0, max_index]
    print('resnet18 max index : {} , value : {}, class name : {}'.format(max_index, max_value, class_name[max_index] ))


if __name__ == '__main__':
    main()

# base model
# device = "cpu:0" 일 때
# 100 iteration time : 2.7593486309051514 [sec]
# device = "gpu:0" 일 때
# 100 iteration time : 0.42092013359069824 [sec]

# jit model
# device = "cpu:0" 일 때
# 100 iteration time : 2.768479824066162 [sec]
# device = "gpu:0" 일 때
# 100 iteration time : 0.36458611488342285 [sec]

# onnx
# device = "cpu" 일 때
# 100 iteration time : 1.2189843654632568 [sec]
# device = "gpu" 일 때
# 100 iteration time : 0.3168525695800781 [sec]