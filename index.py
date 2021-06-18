import test
file = ["Neural Network/MNIST/mnist10x20.nnet", "Neural Network/MNIST/mnist20x20.nnet", "Neural Network/MNIST/mnist20x40.nnet"]
file_Auto_Taxi = ["Neural Network/AutoTaxi/AutoTaxi_32Relus_200Epochs_OneOutput.nnet", "Neural Network/AutoTaxi/AutoTaxi_64Relus_200Epochs_OneOutput.nnet", "Neural Network/AutoTaxi/AutoTaxi_128Relus_200Epochs_OneOutput.nnet"]
# dims = [[784, 20, 10], [784, 50, 10], [784, 50, 50, 10], [784, 100, 100, 10]]
dims = [[10,20,10]]

for item in file_Auto_Taxi:
    test.test(0.1, file_path=item)