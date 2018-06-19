#!/usr/bin/python
import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tensorflow.contrib.framework import arg_scope
import numpy as np
import os
resolution = 256
model_name= '256_256_resfcn256_weight'
model_path = '256_256_resfcn256_weight.data-00000-of-00001'
#all_vars = []
def resBlock(x, num_outputs, kernel_size = 4, stride=1, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, scope=None):
    assert num_outputs%2==0 #num_outputs must be divided by channel_factor(2 here)
    with tf.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_outputs:
            shortcut = tcl.conv2d(shortcut, num_outputs, kernel_size=1, stride=stride,
                                  activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=1, stride=1, padding='SAME')
        x = tcl.conv2d(x, num_outputs/2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = tcl.conv2d(x, num_outputs, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)
        
        x += shortcut
        x = normalizer_fn(x)
    x = activation_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_inp = 256, resolution_op = 256, channel = 3, name = 'resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
    
    def __call__(self, x, is_training = False):
        with tf.variable_scope(self.name) as scope:
            with arg_scope([tcl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([tcl.conv2d, tcl.conv2d_transpose], activation_fn=tf.nn.relu,
                               normalizer_fn=tcl.batch_norm,
                               biases_initializer=None,
                               padding='SAME',
                               weights_regularizer=tcl.l2_regularizer(0.0002)):
                    size = 16
                    # x: s x s x 3
                    se = tcl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) # 256 x 256 x 16
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=2) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 2, kernel_size=4, stride=1) # 128 x 128 x 32
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=2) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 4, kernel_size=4, stride=1) # 64 x 64 x 64
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=2) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 8, kernel_size=4, stride=1) # 32 x 32 x 128
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=2) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 16, kernel_size=4, stride=1) # 16 x 16 x 256
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=2) # 8 x 8 x 512
                    se = resBlock(se, num_outputs=size * 32, kernel_size=4, stride=1) # 8 x 8 x 512
                    
                    pd = tcl.conv2d_transpose(se, size * 32, 4, stride=1) # 8 x 8 x 512
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=2) # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 16, 4, stride=1) # 16 x 16 x 256
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=2) # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 8, 4, stride=1) # 32 x 32 x 128
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=2) # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64
                    pd = tcl.conv2d_transpose(pd, size * 4, 4, stride=1) # 64 x 64 x 64
                    
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=2) # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size * 2, 4, stride=1) # 128 x 128 x 32
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=2) # 256 x 256 x 16
                    pd = tcl.conv2d_transpose(pd, size, 4, stride=1) # 256 x 256 x 16
                    
                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                    pd = tcl.conv2d_transpose(pd, 3, 4, stride=1) # 256 x 256 x 3
                    pos = tcl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn = tf.nn.sigmoid)#, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
       	            return pos
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class PosPrediction():
    def __init__(self, resolution_inp = 256, resolution_op = 256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp*1.1       #281.6
        
        # network type
        self.network = resfcn256(self.resolution_inp, self.resolution_op)
        
        # net forward
        self.x = tf.placeholder(tf.float32, shape=[None, self.resolution_inp, self.resolution_inp, 3])
        self.x_op = self.network(self.x, is_training = False)
        self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        #self.sess=tf.Session()
	#self.sess.run(tf.global_variables_initializer())

    def restore(self, model_path):
        #all_vars = self.network.vars
	tf.train.Saver(self.network.vars).restore(self.sess, model_path)

def predict(self, image):
    pos = self.sess.run(self.x_op,
                        feed_dict = {self.x: image[np.newaxis, :,:,:]})
    pos = np.squeeze(pos)
    return pos*self.MaxPos                 #why?

def predict_batch(self, images):
    pos = self.sess.run(self.x_op,
                        feed_dict = {self.x: images})
    return pos*self.MaxPos

def main():
    
	#sess.run(tf.initialize_all_variables())
        pos_predictor = PosPrediction(resolution, resolution)
        if not os.path.isfile(model_path):
            print("please download PRN trained model first.")
            exit()
	sess = pos_predictor.sess
        pos_predictor.restore(model_name)
        for var in tf.trainable_variables():
            print var.name
        #pos_predictor.restore(model_name)
        #new_saver.restore(sess, tf.train.latest_checkpoint('256_256_resfcn256_weight.data-00000-of-00001'))
        all_vars = tf.trainable_variables()
        for v in all_vars:
            name = v.name
            fname = name + '.prototxt'
            fname = fname.replace('/','_')
            print fname
            v_4d = np.array(sess.run(v))
            if v_4d.ndim == 4:
                #v_4d.shape [ H, W, I, O ]
                v_4d = np.swapaxes(v_4d, 0, 2) # swap input and output
                v_4d = np.swapaxes(v_4d, 1, 3) # swap W, O
                v_4d = np.swapaxes(v_4d, 0, 1) # swap I, O
                v_4d.shape [ O, I, H, W ]
                f = open(fname, 'w')
                vshape = v_4d.shape[:]
                v_1d = v_4d.reshape(v_4d.shape[0]*v_4d.shape[1]*v_4d.shape[2]*v_4d.shape[3])
                f.write('  blobs {\n')
                for vv in v_1d:
                    f.write('    data: %8f' % vv)
                    f.write('\n')
                f.write('    shape {\n')
                for s in vshape:
                    f.write('      dim: ' + str(s))#print dims
                    f.write('\n')
                f.write('    }\n')
                f.write('  }\n')
            elif v_4d.ndim == 2:
                print("------2dim------")
            elif v_4d.ndim == 3:
                print("------3dim------")
            elif v_4d.ndim == 1 :#do not swap
                f = open(fname, 'w')
                f.write('  blobs {\n')
                for vv in v_4d:
                    f.write('    data: %.8f' % vv)
                    f.write('\n')
                f.write('    shape {\n')
                f.write('      dim: ' + str(v_4d.shape[0]))#print dims
                f.write('\n')
                f.write('    }\n')
                f.write('  }\n')
            f.close()

if __name__ == '__main__':
    main()
