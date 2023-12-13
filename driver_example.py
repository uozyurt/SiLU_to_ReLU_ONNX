from replace_SiLU_layers import replace_silu




model_input_path = "yolov8n.onnx"
model_output_path = "yolov8n_modified_ReLU.onnx"
new_activation_name = "Relu" #onnx name for ReLU (can be changed with any other activation functions in onnx which don't have parameters)

replace_silu(model_input_path, model_output_path, new_activation_name)


model_input_path = "yolov8n.onnx"
model_output_path = "yolov8n_modified_LeakyReLU.onnx"
new_activation_name = "LeakyRelu" # onnx name for LeakyReLU
alpha_for_LeakyRelu = 1 # example negative slope

replace_silu(model_input_path, model_output_path, new_activation_name, alpha_for_LeakyRelu)



