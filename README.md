# OpenVINO_EX
- CPU 11th Gen Intel(R) Core(TM) i7-11375H @ 3.30GHz (cpu)
- Intel(R) Iris(R) Xe Graphics (iGPU)
- NVIDIA GeForce RTX 3060 Laptop GPU (gpu)

# 1.OpenVINO_Test
- openVINO hello world example

# 2.OpenVINO_Benchmark 
- resnet18 Model
- Performace evaluation(Execution time of 100 iteration for one 224x224x3 image)
- (Calculated on the 2021-12-05)

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>cpu</td>
			<td><strong>PyTorch</strong></td>
            <td><strong>PyTorch JIT</strong></td>
            <td><strong>ONNX Runtime</strong></td>
            <td><strong>openVINO</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>37.61 ms</td>
			<td>36.07 ms</td>
			<td>16.66 ms</td>
			<td>11.76 ms</td>
		</tr>
	</tbody>
</table>

<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>gpu</td>
            <td><strong>PyTorch</strong></td>
            <td><strong>PyTorch JIT</strong></td>
            <td><strong>openVINO (igpu)</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>5.01 ms</td>
			<td>4.04 ms</td>
			<td>5.30 ms</td>
		</tr>
	</tbody>
</table>

# 3.OpenVINO_PTQ
- resnet18 Model
- Performace evaluation(Execution time of 100 iteration for one 224x224x3 image)
- Using 1000 images from Imagenet 2017 validation dataset for calibration
- Post train qauntization
- (Calculated on the 2021-12-05)
<table border="0"  width="100%">
	<tbody align="center">
		<tr>
			<td>cpu</td>
			<td><strong>openVINO</strong></td>
            <td><strong>openVINO</strong></td>
            <td><strong>ONNX Runtime</strong></td>
            <td><strong>openVINO</strong></td>
		</tr>
		<tr>
			<td>precision</td>
			<td><strong>f32</strong></td>
            <td><strong>int8</strong></td>
            <td><strong>f32</strong></td>
            <td><strong>int8</strong></td>
		</tr>
		<tr>
			<td>Avg Duration time [ms]</td>
			<td>37.61 ms</td>
			<td>36.07 ms</td>
			<td>16.66 ms</td>
			<td>11.76 ms</td>
		</tr>
	</tbody>
</table>


# 4.OpenVINO QAT (준비중)

# 5.OpenVINO Custom Operations (준비중)

#Reference
- openVINO Tutorials : <https://docs.openvino.ai/latest/tutorials.html>
- ONNX Runtime Tutorials : <https://onnxruntime.ai/docs/tutorials/>