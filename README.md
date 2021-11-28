# OpenVINO_EX
- CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz
- Intel(R) Iris(R) Xe Graphics (iGPU)
- NVIDIA GeForce RTX 3060 Laptop GPU


# 1.OpenVINO_Test
- openVINO hello world example


# 2.OpenVINO_Benchmark
- 224x224x3 이미지 1개 100회 반복 계산 수행시간 비교
* PyTorch (cpu) -> 2759 [ms]
* PyTorch (gpu) -> 389 [ms]
* PyTorch JIT (cpu) -> 2768 [ms]
* PyTorch JIT (gpu) -> 364 [ms]
* ONNX Runtime (cpu) -> 1218 [ms]
* ONNX Runtime (gpu) -> 316 [ms]
* openVINO (cpu) -> 1110 [ms]
* openVINO (igpu) -> 510 [ms]


# 3.OpenVINO_QAT (진행중)


# 4.OpenVINO Custom nGraph Operations (준비중)


#Reference
- openVINO Tutorials : <https://docs.openvino.ai/latest/tutorials.html>
- ONNX Runtime Tutorials : <https://onnxruntime.ai/docs/tutorials/>