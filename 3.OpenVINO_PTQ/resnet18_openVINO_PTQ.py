import torch, torchvision, cv2, struct, time
import numpy as np
from utils import *
import torch.onnx
import onnx
from openvino.inference_engine import IECore
from addict import Dict
from pathlib import Path
import copy
import os
from compression.api import DataLoader, Metric
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline

print("onnx:", onnx.__version__)
ie = IECore()
for device in ie.available_devices:
    device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

#intel_device = 'GPU'
intel_device = 'CPU'
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

    if intel_device == 'CPU' :
        onnx_model_name =  "model/resnet18_cpu.onnx"
    else:
        onnx_model_name =  "model/resnet18_gpu.onnx"

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

    ir_path = onnx_model_name.split('.')[0] + (".xml")
    ir_w_path = onnx_model_name.split('.')[0] + (".bin")
    if not os.path.isfile(ir_path):
        print("Model Optimizer command to convert the ONNX model to OpenVINO:")
        mo_command = f"""mo
                         --input_model "{onnx_model_name}"
                         --input_shape "[1,3, {224}, {224}]"
                         --data_type FP16
                         --output_dir "model"
                         """
        mo_command = " ".join(mo_command.split())
        print("Exporting ONNX model to IR... This may take a few minutes.")
        print(mo_command)
        os.system(mo_command)
    else:
        print(f"IR model {ir_path} already exists.")

    if intel_device == 'CPU' :
        compressed_model_xml = "model/optimized/resnet18_cpu.xml"
    else:
        compressed_model_xml = "model/optimized/resnet18_gpu.xml"

    if not os.path.isfile(compressed_model_xml):
        model_config = Dict(
            {
                "model_name": "resnet18",
                "model": f"{ir_path}",
                "weights": f"{ir_w_path}",
            }
        )

        engine_config = Dict({"device": f"{intel_device}", "stat_requests_number": 2, "eval_requests_number": 2})

        algorithms = [
            {
                "name": "DefaultQuantization",
                "params": {
                    "target_device": f"{intel_device}",
                    "preset": "performance",
                    "stat_subset_size": 1000,
                },
            }
        ]

        # Imagenet 2017 validation dataset
        dataset_path = 'F:\dataset\imagenet_dataset\ILSVRC2012_img_val'
        data_dir = Path(dataset_path)

        class ClassificationDataLoader(DataLoader):
            """
            DataLoader for image data that is stored in a directory per category. For example, for
            categories _rose_ and _daisy_, rose images are expected in data_source/rose, daisy images
            in data_source/daisy.
            """

            def __init__(self, data_source):
                """
                :param data_source: path to data directory
                """
                self.data_source = Path(data_source)
                self.dataset = [p for p in data_dir.glob("**/*") if p.suffix in (".png", ".jpg", ".JPEG")]
                self.class_names = sorted([item.name for item in Path(data_dir).iterdir() if item.is_dir()])

            def __len__(self):
                """
                Returns the number of elements in the dataset
                """
                return len(self.dataset)

            def __getitem__(self, index):
                """
                Get item from self.dataset at the specified index.
                Returns (annotation, image), where annotation is a tuple (index, class_index)
                and image a preprocessed image in network shape
                """
                if index >= len(self):
                    raise IndexError
                filepath = self.dataset[index]
                annotation = (index, self.class_names.index(filepath.parent.name))
                image = self._read_image(filepath)
                return annotation, image

            def _read_image(self, index):
                """
                Read image at dataset[index] to memory, resize, convert to BGR and to network shape

                :param index: dataset index to read
                :return ndarray representation of image batch
                """
                image = cv2.imread(os.path.join(self.data_source, index))
                image = cv2.resize(image, (224, 224))
                img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr -> rgb
                img3 = img2.transpose(2, 0, 1)  # hwc -> chw
                img4 = img3.astype(np.float32)  # uint -> float32
                img4 /= 255  # 1/255
                img5 = torch.from_numpy(img4)  # numpy -> tensor
                if half:  # f32 -> f16
                    img5 = img5.half()
                img6 = img5.unsqueeze(0)  # [c,h,w] -> [1,c,h,w]

                return img6

        class Accuracy(Metric):
            def __init__(self):
                super().__init__()
                self._name = "accuracy"
                self._matches = []

            @property
            def value(self):
                """Returns accuracy metric value for the last model output."""
                return {self._name: self._matches[-1]}

            @property
            def avg_value(self):
                """
                Returns accuracy metric value for all model outputs. Results per image are stored in
                self._matches, where True means a correct prediction and False a wrong prediction.
                Accuracy is computed as the number of correct predictions divided by the total
                number of predictions.
                """
                num_correct = np.count_nonzero(self._matches)
                return {self._name: num_correct / len(self._matches)}

            def update(self, output, target):
                """Updates prediction matches.

                :param output: model output
                :param target: annotations
                """
                predict = np.argmax(output[0], axis=1)
                match = predict == target
                self._matches.append(match)

            def reset(self):
                """
                Resets the Accuracy metric. This is a required method that should initialize all
                attributes to their initial value.
                """
                self._matches = []

            def get_attributes(self):
                """
                Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
                Required attributes: 'direction': 'higher-better' or 'higher-worse'
                                     'type': metric type
                """
                return {self._name: {"direction": "higher-better", "type": "accuracy"}}

        # Step 1: Load the model
        model = load_model(model_config=model_config)
        original_model = copy.deepcopy(model)

        # Step 2: Initialize the data loader
        data_loader = ClassificationDataLoader(data_source=data_dir)

        # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric
        #        Compute metric results on original model
        metric = Accuracy()

        # Step 4: Initialize the engine for metric calculation and statistics collection
        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=metric)

        # Step 5: Create a pipeline of compression algorithms
        pipeline = create_pipeline(algo_config=algorithms, engine=engine)

        # Step 6: Execute the pipeline
        compressed_model = pipeline.run(model=model)

        # Step 7 (Optional): Compress model weights quantized precision
        #                    in order to reduce the size of final .bin file
        compress_model_weights(model=compressed_model)

        # Step 8: Save the compressed model and get the path to the model
        compressed_model_paths = save_model(
            model=compressed_model, save_path=os.path.join(os.path.curdir, "model/optimized")
        )
        compressed_model_xml = Path(compressed_model_paths[0]["model"])
        print(f"The quantized model is stored in {compressed_model_xml}")
    else:
        print(f"optimized IR model {compressed_model_xml} already exists.")

    # # Step 9 (Optional): Evaluate the original and compressed model. Print the results
    # original_metric_results = pipeline.evaluate(original_model)
    # if original_metric_results:
    #     print(f"Accuracy of the original model:  {next(iter(original_metric_results.values())):.5f}")

    # quantized_metric_results = pipeline.evaluate(compressed_model)
    # if quantized_metric_results:
    #     print(f"Accuracy of the quantized model: {next(iter(quantized_metric_results.values())):.5f}")

    net_onnx = ie.read_network(model=compressed_model_xml)
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

# base model 2021-12-05
# device = "cpu:0" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 3.76161527633667 [sec]
# device = "gpu:0" 일 때 (NVIDIA GeForce RTX 3060 Laptop GPU)
# 100 iteration time : 0.501741886138916 [sec]

# jit model 2021-12-05
# device = "cpu:0" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 3.6070895195007324 [sec]
# device = "gpu:0" 일 때 (NVIDIA GeForce RTX 3060 Laptop GPU)
# 100 iteration time : 0.4049530029296875 [sec]

# onnx model 2021-12-05
# device = "cpu" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 1.6663672924041748 [sec]
# device = "gpu" 일 때 (NVIDIA GeForce RTX 3060 Laptop GPU)
# 100 iteration time : 1.1880877017974854 [sec]

# openVINO(f32) 2021-12-05
# device = "cpu" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz)
# 100 iteration time : 1.176347255706787 [sec]
# device = "gpu" 일 때 (Intel(R) Iris(R) Xe Graphics (iGPU))
# 100 iteration time : 0.5303637981414795 [sec]

# device = "cpu" 일 때 (11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz) PTQ
# 100 iteration time : 0.3785285949707031 [sec]
# device = "gpu" 일 때 (Intel(R) Iris(R) Xe Graphics (iGPU))
# 100 iteration time : 0.28206872940063477 [sec]