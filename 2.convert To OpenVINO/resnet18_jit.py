import torch, torchvision, os, cv2, struct, time
import numpy as np
from utils import *

print('gpu device count : ', torch.cuda.device_count())
print('device_name : ', torch.cuda.get_device_name(0))
print('gpu available : ', torch.cuda.is_available())

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
device = "cpu:0"
print(f"{device}")

def main():

    if os.path.isfile('model/resnet18_jit.pth'):                              # resnet18_jit.pth 파일이 있다면
        net = torch.jit.load('model/resnet18_jit.pth', map_location=device)   # resnet18_jit.pth 파일 로드
    else:                                                               # resnet18_jit.pth 파일이 없다면
        with torch.no_grad():
            if os.path.isfile('model/resnet18.pth'):                          # resnet18.pth 파일이 있다면
                net = torch.load('model/resnet18.pth')                        # resnet18.pth 파일 로드
            else:  # resnet18.pth 파일이 없다면
                net = torchvision.models.resnet18(pretrained=True)      # torchvision에서 resnet18 pretrained weight 다운
                torch.save(net, 'model/resnet18.pth')                         # resnet18.pth 파일 저장
            torch.jit.save(torch.jit.script(net), 'model/resnet18_jit.pth')   # resnet18_jit.pth 파일 저장
        net = torch.jit.load('model/resnet18_jit.pth', map_location=device)  # resnet18_jit.pth 파일 로드

    #half = True
    half = False
    net = net.eval()                            # 모델을 평가 모드로 세팅
    net = net.to(device)                        # gpu 설정
    if half:
        net.half()  # to FP16

    img = cv2.imread('date/panda0.jpg')  # image file load
    dur_time = 0
    iteration = 100

    # 속도 측정에서 첫 1회 연산 제외하기 위한 계산
    out = infer(img, net, half, )
    torch.cuda.synchronize()

    for i in range(iteration):
        begin = time.time()
        out = infer(img, net, half, device)
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