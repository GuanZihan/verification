import test_taxi
file = ["Neural Network/MNIST/mnist10x20.nnet", "Neural Network/MNIST/mnist20x20.nnet", "Neural Network/MNIST/mnist20x40.nnet"]
file_Auto_Taxi = ["Neural Network/AutoTaxi/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"]
# dims = [[784, 20, 10], [784, 50, 10], [784, 50, 50, 10], [784, 100, 100, 10]]
dims = [[10,20,10]]
epss = [0.04, 0.08, 0.016]
for eps in epss:
    for item in file_Auto_Taxi:
        test_taxi.test(eps, file_path=item)