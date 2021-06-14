import tensorflow as tf
import matlab.engine
import numpy as np
import scipy.io as scio
import Utils.Utils as util
from src.NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
eps = 0.1

# ===================
# weights = 0
# bias = 0
# with util.open_file("models/raghunathan18_lpnn.pkl", 'rb') as f:
#     file = np.load(f, allow_pickle=True)
#     W = []
#     b = []
#     for layer in file:
#         W.append(layer[0].T)
#         b.append(np.expand_dims(layer[1], axis=1))
#     weights = W
#     bias = b
# dims = [784, 20, 10]
# =====================
# 0. for reading nnet file
# dims, weights, bias, x_min, x_max = util.read_nn("Neural Network/ACASXu/ACASXU_experimental_v2a_1_2.nnet")
# dims = [784, 200, 100, 50, 10]
dims = [784, 5, 10]
print(dims)
nn = NeuralNetwork(dims)
# 1. for aditi load_weights
# nn.load_weights()

# 2. for training
nn.train(cache=False)
nn.read_weights()

# 3. for random NN
# nn.train(cache=True)
# nn.generateRandomWeights()

# start matlab
eng = matlab.engine.start_matlab()
eng.cd("D:\WORK_SPACE\DeepSDP\DeepSDP")
eng.addpath(r'matlab_engine')

# save weights and bias
util.write_single_data_to_matlab_path('D:\WORK_SPACE\DeepSDP\DeepSDP\weights.mat', "weights", np.array(nn.weights))
util.write_single_data_to_matlab_path('D:\WORK_SPACE\DeepSDP\DeepSDP\ias.mat', 'bias', np.array(nn.bias))

# for every samples
for i in range(1):
    sample_image = test_images[i] / 255
    sample_image = sample_image.reshape((784, 1))

    # generate a random sample
    # np.random.seed(i)
    # sample_image = np.random.rand(dims[0], 1)
    # sample_image = util.read_sample("Dataset/AutoTaxi/AutoTaxi_ExampleImage.npy")

    # save sample
    util.write_single_data_to_matlab_path('D:\WORK_SPACE\DeepSDP\DeepSDP\sample.mat', 'input', sample_image)

    # convert dims to a matlab.double data structure
    dims_double = matlab.double(dims)

    # sample_image = matlab.double(sample_image.T.tolist())
    # sample_label = 3
    sample_label = test_labels.tolist()[i]

    res = eng.test_mnist(eps, 1, dims_double, sample_label + 1, nargout=1)

    a = scio.loadmat("D:\WORK_SPACE\DeepSDP\DeepSDP\optimal.mat")
    pre = a["out_op"][1: 785]
    plt.imshow(pre.reshape(28, 28))
    plt.show()
    plt.imshow(test_images[i])
    plt.show()
    nn.predict_manual(pre)
    pred_label, _ = nn.predict(pre)
    print(pred_label)
    print("index " + str(i) + " sample label is " + str(sample_label) + " optimal value " + str(res))