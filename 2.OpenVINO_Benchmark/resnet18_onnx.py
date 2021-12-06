import torch, torchvision, os, cv2, struct, time
import numpy as np
from utils import *
import torch.onnx
import onnx
import onnxruntime
import psutil

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    print('gpu device count : ', torch.cuda.device_count())
    print('device_name : ', torch.cuda.get_device_name(0))
    print('torch gpu available : ', torch.cuda.is_available())

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

    sess_options = onnxruntime.SessionOptions()
    sess_options.optimized_model_filepath = "model/onnx_resnet18_{}.onnx".format(device.type)
    ort_session = onnxruntime.InferenceSession(onnx_model_name, sess_options)

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

# base model 2021-12-06
# device = "cpu:0" 일 때
# 100 iteration time : 3.235487461090088 [sec]
# device = "gpu:0" 일 때
# 100 iteration time : 0.3634309768676758 [sec]

# jit model 2021-12-06
# device = "cpu:0" 일 때
# 100 iteration time : 2.554605007171631 [sec]
# device = "gpu:0" 일 때
# 100 iteration time : 0.34999537467956543 [sec]

# onnx model
# device = "cpu" 일 때
# 100 iteration time : 1.2032380104064941 [sec]
# device = "gpu" 일 때
# 100 iteration time : 0.9713826179504395 [sec]