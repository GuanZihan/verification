import h5py

f = h5py.File('D:/WORK_SPACE/acasxu_tf_keras/acasxu_tf_keras/ACASXU_experimental_v2a_1_1..h5','r')
for key, value in f.attrs.items():
    print("  {}: {}".format(key, value))