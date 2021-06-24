import matlab.engine
import numpy as np
import Utils.Utils as util
import os
from src.NeuralNetwork import NeuralNetwork


def test(eps, file_path=""):
    """
    test
    :param eps:
    :param file_path:
    :return:
    """

    # 0. for reading nnet file
    dims, weights, bias, x_min, x_max = util.read_nn(file_path)
    print("model: ", file_path)
    print("dims: ", dims)
    nn = NeuralNetwork(dims)
    # 1. for aditi load_weights
    # nn.load_weights()

    # 2. for training
    # nn.train(cache=False, mode=2)
    # nn.read_weights()

    # 3. for random NN
    # nn.train(cache=True)
    # nn.generateRandomWeights()

    # 4. for adding weights and bias to the instance
    nn.weights = weights
    nn.bias = bias
    nn.dims = dims

    # start matlab
    eng = matlab.engine.start_matlab()
    eng.cd(r"matlab")

    # save weights and bias
    util.write_single_data_to_matlab_path('./matlab/weights.mat', "weights", nn.weights)
    util.write_single_data_to_matlab_path('./matlab/ias.mat', 'bias', nn.bias)

    solved_primal = 0
    solved_dual = 0
    solved_plus = 0

    # for root,dirs,files in os.walk("./Dataset/"):
    #     sample_file_path = files

    for i in range(1):
        # sample_image = test_images[i] / 255
        # np.random.seed(i)
        # sample_image = np.random.rand(dims[0], 1) / 10
        sample_image = np.array([[0.64, -0.2, -0.2, 0.47, -0.47]]).T

        # sample_image = sample_image.reshape((-1, 1))
        # print(sample_image)
        # break
        # plt.imshow(sample_image)
        # plt.show()

        # generate a random sample
        # np.random.seed(i)
        # sample_image = np.random.rand(dims[0], 1)
        # sample_image = util.read_sample("Dataset/AutoTaxi/AutoTaxi_ExampleImage.npy")

        # save sample
        util.write_single_data_to_matlab_path('./matlab/sample.mat', 'input', sample_image)

        # convert dims to a matlab.double data structure
        dims_double = matlab.double(dims)

        # sample_image = matlab.double(sample_image.T.tolist())

        pred = nn.predict_manual_mnist(sample_image)
        sample_label = int(pred[1])

        # SDR
        res_primal = eng.test_acas(eps, 1, dims_double, sample_label + 1, 1, nargout=3)

        # DeepSDP
        res_dual = eng.test_acas(eps, 1, dims_double, sample_label + 1, 2, nargout=3)

        # Deeplus
        res_plus = eng.test_acas(eps, 1, dims_double, sample_label + 1, 3, nargout=3)

        if res_primal[2] == 1.0:
            solved_primal += 1

        if res_dual[2] == 1.0:
            solved_dual += 1

        if res_plus[2] == 1.0:
            solved_plus += 1

        # a = scio.loadmat("D:\WORK_SPACE\DeepSDP\DeepSDP\optimal.mat")
        # pre = a["out_op"][1: 785]
        # plt.imshow(pre.reshape(28, 28))
        # plt.savefig(str(file_path) + "_adversarial_" + str(i) + ".png")
        # pred = nn.predict_manual(pre)

        # if res[0] > 0 and pred[1] == sample_label:
        #     false_negative = 1
        ret = {
            "model_name": file_path,
            "Primal": res_primal[0],
            "Primal_time": res_primal[1],
            "Dual": res_dual[0],
            "Dual_time": res_dual[1],
            "status_primal": res_primal[2],
            "status_dual": res_dual[2],
            "res_plus": res_plus[0],
            "res_plus_time": res_plus[1],
            "status_res_plus": res_plus[2],
        }

        with open(str(file_path) + "_log.txt", "a+") as f:
            f.write(str(ret))
            f.write("\n")

    with open(str(file_path) + "_log.txt", "a+") as f:
        f.write("primal solved number: " + str(solved_primal))
        f.write("\n")
        f.write("Dual solved number: " + str(solved_dual))
        f.write("\n")
        f.write("Plus solved number: " + str(solved_plus))
        f.write("\n")


test(0.0008, "D:/WORK_SPACE/verification/Neural Network/ACASXu/ACASXU_experimental_v2a_1_3.nnet")