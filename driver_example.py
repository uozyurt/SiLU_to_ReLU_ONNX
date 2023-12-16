from replace_SiLU_layers import replace_silu




model_input_path = "yolov8s.onnx"
model_output_path = "yolov8s_relu.onnx"
new_activation_name = "Relu" #onnx name for ReLU (can be changed with any other activation functions in onnx which don't have parameters)

replace_silu(model_input_path, model_output_path, new_activation_name)


model_input_path = "yolov8s.onnx"
model_output_path = "yolov8s_leakyrelu.onnx"
new_activation_name = "LeakyRelu" # onnx name for LeakyReLU
alpha_for_LeakyRelu = 0.2 # example negative slope

replace_silu(model_input_path, model_output_path, new_activation_name, alpha_for_LeakyRelu)



