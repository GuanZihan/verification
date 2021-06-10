import numpy as np
def read_nn(file_name):
    with open(file_name, "r") as f:
        line = f.readline()
        while line.startswith("//"):
            line = f.readline()
        layers, dim_in, dim_out, neuronsize = eval(line)
        dims = list(eval(f.readline()))
        f.readline()  # A flag that is no longer used
        input_min = np.array(list(eval(f.readline())))
        input_max = np.array(list(eval(f.readline())))
        f.readline()
        f.readline()
        weights = []
        bias = []
        for i in dims[1:]:
            current_layer = i
            weights_layer = []
            bias_layer = []
            for j in range(current_layer):
                weights_layer.append(list(eval(f.readline())))
            for j in range(current_layer):
                bias_layer.append(list(eval(f.readline())))
            weights_layer = np.array(weights_layer)
            bias_layer = np.array(bias_layer)
            weights.append(weights_layer)
            bias.append(bias_layer)
        return dims, weights, bias, input_min, input_max



read_nn("ACASXU_experimental_v2a_1_1.nnet")