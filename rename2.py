import os
path = './'
for file in os.listdir(path):
    name = 'Conv2d_transpose_16_BatchNorm_beta.prototxt'
    if file == name:
        new_name = 'Conv2d_transpose_17_BatchNorm_beta.prototxt'
        os.rename(os.path.join(path,name),os.path.join(path,new_name))
    name = 'Conv2d_transpose_16_BatchNorm_gamma.prototxt'
    if file == name:
        new_name = 'Conv2d_transpose_17_BatchNorm_gamma.prototxt'
        os.rename(os.path.join(path,name),os.path.join(path,new_name))
    name = 'Conv2d_transpose_16_weights.prototxt'
    if file == name:
        new_name = 'Conv2d_transpose_17_weights.prototxt'
        os.rename(os.path.join(path,name),os.path.join(path,new_name))
    name = 'Conv2d_transpose_10_BatchNorm_beta.prototxt'
    if file == name:
        new_name = 'Conv2d_transpose_11_BatchNorm_beta.prototxt'
        os.rename(os.path.join(path,name),os.path.join(path,new_name))
    name = 'Conv2d_transpose_10_BatchNorm_gamma.prototxt'
    if file == name:
        new_name = 'Conv2d_transpose_11_BatchNorm_gamma.prototxt'
        os.rename(os.path.join(path,name),os.path.join(path,new_name))
    name = 'Conv2d_transpose_10_weights.prototxt'
    if file == name:
        new_name = 'Conv2d_transpose_11_weights.prototxt'
        os.rename(os.path.join(path,name),os.path.join(path,new_name))


