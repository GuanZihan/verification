import scipy.io as scio
import matlab.engine
a = {1:[3], 2:[5]}
scio.savemat(file_name='D:\WORK_SPACE\DeepSDP\DeepSDP\data.mat', mdict={'p':a, 'a': 123})
eng = matlab.engine.start_matlab()
eng.cd("D:\WORK_SPACE\DeepSDP\DeepSDP")
eng.addpath(r'./')
eng.test()