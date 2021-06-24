import matlab.engine
import numpy as np
import Utils.Utils as util
import matplotlib.pyplot as plt
import os
from src.NeuralNetwork import NeuralNetwork
from tensorflow.keras import datasets

# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


def test(eps, file_path=""):
    """
    test
    :param eps:
    :param file_path:
    :return:
    """

    dims = [20, 500, 10]
    # dims = file_path
    # dims = file_path
    print("model: ", file_path)
    print("dims: ", dims)
    nn = NeuralNetwork(dims)

    nn.generateRandomWeights()

    # start matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r"matlab")
    eng.addpath(r'matlab')

    # save weights and bias
    util.write_single_data_to_matlab_path('./matlab/weights.mat', "weights", nn.weights)
    util.write_single_data_to_matlab_path('./matlab/ias.mat', 'bias', nn.bias)

    solved_primal = 0
    solved_dual = 0
    solved_plus = 0
    for i in range(1):

        # sample_image = sample_image.reshape((-1, 1))
        # print(sample_image)
        # break
        # plt.imshow(sample_image)
        # plt.show()

        # generate a random sample
        # np.random.seed(i)
        np.random.seed(i)
        sample_image = np.random.rand(dims[0], 1) - 0.8

        # sample_image = util.read_sample("Dataset/AutoTaxi/AutoTaxi_ExampleImage.npy")

        # save sample
        util.write_single_data_to_matlab_path('./matlab/sample.mat', 'input', sample_image)

        # convert dims to a matlab.double data structure
        dims_double = matlab.double(dims)

        # sample_image = matlab.double(sample_image.T.tolist())
        # sample_label = 3

        # Tensorflow dataset
        # sample_label = test_labels.tolist()[i]

        sample_label = 0
        pred = nn.predict_manual_taxi(sample_image)

        # SDR
        # res_primal = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 1, nargout=3)
        #
        # # DeepSDP
        # res_dual = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 2, nargout=3)
        #
        # Deeplus
        res_plus = eng.test_auto_taxi(eps, float(pred[0][0]), dims_double, sample_label + 1, 3, nargout=3)

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

test(0.2)