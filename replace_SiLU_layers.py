import onnx
from onnx import helper



def strip_layer(input_layer_name):
    stripped_output = str(input_layer_name)
    stripped_output = stripped_output.replace("[", "")
    stripped_output = stripped_output.replace("]", "")
    stripped_output = stripped_output.replace("'", "")
    stripped_output = stripped_output.replace("\"", "")
    stripped_output = stripped_output.replace(" ", "")
    return stripped_output

def strip_and_split_input(input_layer_name):
    stripped_output = strip_layer(input_layer_name)
    
    split_output = stripped_output.split(",")
    return split_output

def check_if_output_is_sigmoid(output):
    stripped_output = strip_layer(output)
    if (stripped_output.endswith("Sigmoid_output_0")):
        return True
    else:
        return False
    

def find_outputs_output(current_node, model):
    ret = []

    for node in model.graph.node:
        if(strip_layer(current_node.output) in strip_and_split_input(node.input)):
            ret.append(node)

    return ret


def insert_node(new_node, old_node, model):
    # Find the index of the old node
    index = -1
    for i, node in enumerate(model.graph.node):
        if node.name == old_node.name:
            index = i
            break

    if index == -1:
        raise ValueError(f"Node {old_node.name} not found in model")
    
    # find the outputs of output of the old node
    output_output = find_outputs_output(old_node, model)
    
    # change the inputs of output of the old node to the inputs of the new node
    for output in output_output:
        for i, input_name in enumerate(output.input):
            if input_name == old_node.output[0]:
                output.input[i] = new_node.output[0]
                break


    # Insert the new node before the old node
    model.graph.node.insert(index, new_node)

    return model


def replace_silu(model_input_path, model_output_path, new_activation_name, alpha_for_LeakyRelu=0.1):

    # load the ONNX model
    model = onnx.load(model_input_path)

    # will be used to name new activation layers
    counter = 0

    # iterate over the nodes
    for node in model.graph.node:
        output_output = find_outputs_output(node, model)

        if(len(output_output) != 2): # this is an assumption about if a SiLU layer will be encountered, then there are 2 nodes (sigmoid and multiplication) connected to output
            continue # if certainly not SiLU, continue

        # iterate the connected nodes to the current nodes output
        for ind, output in enumerate(output_output):
            if (check_if_output_is_sigmoid(output.output)): # check if any sigmoid layer exists

                sigmoid_node = output

                if (ind == 0):
                    mul_node = output_output[1]
                elif (ind == 1):
                    mul_node = output_output[0]

                if(mul_node.op_type != "Mul"): # if the other node is not a multiplication, continue
                    continue

                print(f"The node with the output: {node.output} is connected to a sigmoid and a multiplication node---->\connected sigmoid nodes output: {sigmoid_node.output}\nconnected multiplication nodes output: {mul_node.output}\n")


                if (new_activation_name == "LeakyRelu"):
                    # create a new LeakyReLU node
                    new_activation_node = helper.make_node(
                        op_type= new_activation_name,
                        inputs=[sigmoid_node.input[0]],
                        outputs=[f"{new_activation_name}_{counter}"],
                        alpha=0.1,
                        name=f"{new_activation_name}_{counter}")
                else:
                    # create a new node (without declaring parameter, suitable for ReLU)
                    new_activation_node = helper.make_node(
                        op_type= new_activation_name,
                        inputs=[sigmoid_node.input[0]],
                        outputs=[f"{new_activation_name}_{counter}"],
                        name=f"{new_activation_name}_{counter}")

                
                # custom function to insert an node in the place of another node, making the output connections
                insert_node(new_activation_node, mul_node, model)

                # remove old nodes
                model.graph.node.remove(sigmoid_node)
                model.graph.node.remove(mul_node)

                counter += 1

                print(f"Removed sigmoid and multiplication (SiLU components), added the {new_activation_name}_{counter} layer.\n\n\n")
                

                


    print("-"*80)
    print("PRINTING THE NEW MODEL:\n")
    # print new model
    for node in model.graph.node:
        print(node.output)

    print("-"*80)
    print("\n\n\n\n")

    # check the models consistency (if fails, model should not work as expected or gives error in run-time)
    onnx.checker.check_model(model)


    # save the modified model
    modified_model_path = model_output_path  # Path for the modified model
    onnx.save(model, modified_model_path)

    # return the model 
    return modified_model_path




                
