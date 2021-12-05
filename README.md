# OpenVINO_EX
- CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)
- Intel(R) Iris(R) Xe Graphics (iGPU)
- NVIDIA GeForce RTX 3060 Laptop GPU (gpu)


# 1.OpenVINO_Test
- openVINO hello world example


# 2.OpenVINO_Benchmark 
- resnet18 Model
- Performace evaluation(Execution time of 100 iteration for one 224x224x3 image)
  - PyTorch (cpu) -> 3761 [ms]
  - PyTorch (gpu) -> 501 [ms]
  * PyTorch JIT (cpu) -> 3607 [ms]
  * PyTorch JIT (gpu) -> 404 [ms]
  * ONNX Runtime (cpu) -> 1666 [ms]
  * openVINO (cpu) -> 1176 [ms]
  * openVINO (igpu) -> 530 [ms]
  - (Calculated on the 2021-12-05)

# 3.OpenVINO_PTQ
- resnet18 Model
- Performace evaluation(Execution time of 100 iteration for one 224x224x3 image)
- Using 1000 images from Imagenet 2017 validation dataset for calibration
  - openVINO (cpu) -> 1176 [ms] (f32)
  - openVINO (cpu) -> 378 [ms] (int8 PTQ)
  - openVINO (igpu) -> 530 [ms] (f32)
  - openVINO (igpu) -> 282 [ms] (int8 PTQ)
  - (Calculated on the 2021-12-05)
  - 
# 4.OpenVINO Custom nGraph Operations (준비중)


#Reference
- openVINO Tutorials : <https://docs.openvino.ai/latest/tutorials.html>
- ONNX Runtime Tutorials : <https://onnxruntime.ai/docs/tutorials/>