import matlab.engine
import numpy as np
import Utils.Utils as util
import matplotlib.pyplot as plt
import os
from src.NeuralNetwork import NeuralNetwork
import crown
from tensorflow.keras import datasets

# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


def test(eps, file_path=""):
    """
    test
    :param eps:
    :param file_path:
    :return:
    """

    with util.open_file("models/raghunathan18_pgdnn.pkl", 'rb') as f:
        file = np.load(f, allow_pickle=True)
        W = []
        b = []
        for layer in file:
            W.append(layer[0].T)
            b.append(np.expand_dims(layer[1], axis=1))
        weights = W
        bias = b
    dims = [784, 200, 100, 50, 10]
    # dims = file_path
    # dims = file_path
    print("model: ", file_path)
    print("dims: ", dims)
    nn = NeuralNetwork(dims)
    # nn.train(cache=True, mode=1)
    # nn.read_weights()

    # nn.generateRandomWeights()

    # start matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r"matlab")
    eng.addpath(r'matlab')

    # save weights and bias
    util.write_single_data_to_matlab_path('matlab/weights.mat', "weights", weights)
    util.write_single_data_to_matlab_path('matlab/ias.mat', 'bias', bias)

    solved_primal = 0
    solved_dual = 0
    solved_plus = 0

    with util.open_file("mnist/x_test.npy", "rb") as f:
        test_images = np.load(f)

    with util.open_file("mnist/y_test.npy", "rb") as f:
        test_labels = np.load(f)

    count = 0

    for i in range(len(test_images[:100])):

        # sample_image = sample_image.reshape((-1, 1))
        # print(sample_image)
        # break
        # plt.imshow(sample_image)
        # plt.show()

        # generate a random sample
        # np.random.seed(i)
        # sample_image = np.random.rand(dims[0], 1) * 2 - 0.5
        sample_image = test_images[i]
        nn.weights = weights
        nn.bias = bias

        sample_image = np.reshape(sample_image, (784, 1))
        res = nn.predict_manual_mnist(sample_image)

        if res[1] == test_labels[i]:
            count += 1
        else:
            print(i)

        # sample_image = util.read_sample("Dataset/AutoTaxi/AutoTaxi_ExampleImage.npy")

        # save sample
        util.write_single_data_to_matlab_path('matlab/sample.mat', 'input', sample_image)

        # convert dims to a matlab.double data structure
        dims_double = matlab.double(dims)

        # sample_image = matlab.double(sample_image.T.tolist())
        # sample_label = 3

        # Tensorflow dataset
        # sample_label = test_labels.tolist()[i]

        np.random.seed(i)
        sample_label = np.random.randint(0, 10)
        pred = nn.predict_manual_taxi(sample_image)

        bias_ex = []
        for j in range(len(bias)):
            bias_ex.append(np.squeeze(bias[j], axis=1))

        print(i, sample_label, test_labels[i])

        ret = crown.compute_worst_bound(weights, bias_ex, test_labels[i], 1, sample_image, pred, 4, "i", 0.1, "ours", "disable", False, False, False, False, "relu")
        print(ret[1][0].shape)

        nn.weights = weights
        nn.bias = bias
        UB_N0 = np.minimum(sample_image + eps, 1)

        LB_N0 = np.maximum(sample_image - eps, 0)
        res = nn.interval_arithmetic(LB_N0, UB_N0, "SDR")
        print(res[1][1])

        util.write_single_data_to_matlab_path('matlab/y_min.mat', 'y_min', ret[0])
        util.write_single_data_to_matlab_path('matlab/y_max.mat', 'y_max', ret[1])
        util.write_single_data_to_matlab_path('matlab/x_min.mat', 'x_min', ret[2])
        util.write_single_data_to_matlab_path('matlab/x_max.mat', 'x_max', ret[3])

        # SDR
        # res_primal = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 1, nargout=3)
        #
        # # DeepSDP
        # res_dual = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 2, nargout=3)
        #
        # Deeplus
        res_plus = eng.test_mnist(eps, float(pred[0][0]), dims_double, sample_label + 1, 3, nargout=3)

        print("original value: ", pred[0], end="\n\n")

        # if res_primal[2] == 1.0:
        #     solved_primal += 1

        # if res_dual[2] == 1.0:
        #     solved_dual += 1
        #
        # if res_plus[2] == 1.0:
        #     solved_plus += 1

        # a = scio.loadmat("D:\WORK_SPACE\DeepSDP\DeepSDP\optimal.mat")
        # pre = a["out_op"][1: 785]
        # plt.imshow(pre.reshape(28, 28))
        # plt.savefig(str(file_path) + "_adversarial_" + str(i) + ".png")
        # pred = nn.predict_manual(pre)

        # if res[0] > 0 and pred[1] == sample_label:
        #     false_negative = 1
        # ret = {
        #     "model_name": file_path,
        #     "Primal": res_primal[0],
        #     "Primal_time": res_primal[1],
        #     "Dual": res_dual[0],
        #     "Dual_time": res_dual[1],
        #     "status_primal": res_primal[2],
        #     "status_dual": res_dual[2],
        #     "res_plus": res_plus[0],
        #     "res_plus_time": res_plus[1],
        #     "status_res_plus": res_plus[2],
        #     "pred": pred[0],
        # }
    print(count)

test(0.1)