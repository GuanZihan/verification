import matlab.engine
import numpy as np
import Utils.Utils as util
import matplotlib.pyplot as plt
from src.NeuralNetwork import NeuralNetwork
# from tensorflow.keras import datasets

# (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


def load_training_data(file_path):
    X = np.load(file_path)
    X_norm = X.reshape(784, 1)
    return X_norm


def test(eps, file_path=""):
    """
    test
    :param eps:
    :param file_path:
    :return:
    """
    # ===================
    weights = 0
    bias = 0
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
    # =====================

    # 0. for reading nnet file
    # dims, weights, bias, x_min, x_max = util.read_nn(file_path)
    # dims = [784, 200, 100, 50, 10]
    # dims = file_path
    # dims = file_path
    print("model: ", file_path)
    print("dims: ", dims)
    # nn = NeuralNetwork(dims)
    # 1. for aditi load_weights
    # nn.load_weights()

    # 2. for training
    # nn.train(cache=False)
    # nn.read_weights()

    # 3. for random NN
    # nn.train(cache=True)
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
    # for every samples
    for i in range(1):
        # sample_image = test_images[i] / 255
        sample_image = load_training_data("Dataset/MNIST/MNISTlabel_0_index_2_.npy")
        sample_image = sample_image.reshape((784, 1))
        # plt.imshow(sample_image, cmap='Greys')
        # plt.show()

        # generate a random sample
        # np.random.seed(i)
        # sample_image = np.random.rand(dims[0], 1)
        # sample_image = util.read_sample("Dataset/AutoTaxi/AutoTaxi_ExampleImage.npy")

        # save sample
        util.write_single_data_to_matlab_path('matlab/sample.mat', 'input', sample_image)

        # convert dims to a matlab.double data structure
        dims_double = matlab.double(dims)

        # sample_image = matlab.double(sample_image.T.tolist())
        # sample_label = 3

        # Tensorflow dataset
        # sample_label = test_labels.tolist()[i]

        sample_label = 0

        # SDR
        res_primal = eng.test_mnist(eps, 1, dims_double, sample_label + 1, 1, nargout=3)

        # DeepSDP
        res_dual = eng.test_mnist(eps, 1, dims_double, sample_label + 1, 2, nargout=3)

        if res_primal[2] == 1.0:
            solved_primal += 1

        if res_dual[2] == 1.0:
            solved_dual += 1

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
        }

        with open(str(file_path) + "_log.txt", "a+") as f:
            f.write(str(ret))
            f.write("\n")

    with open(str(file_path) + "_log.txt", "a+") as f:
        f.write("primal solved number: " + str(solved_primal))
        f.write("\n")
        f.write("Dual solved number: " + str(solved_dual))
        f.write("\n")