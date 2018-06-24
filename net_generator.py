import sys
import os
import numpy as np


def generate_conv_layer(name, bottom, top, num_output, kernel_size, pad, stride):
    conv_layer_str = '''layer {
        name: "%s"
        bottom: "%s"
        top: "%s"
        type: "Convolution"
        convolution_param {
        num_output: %s
        kernel_size: %s
        pad: %s
        stride: %s
        }
        \n'''%(name, bottom, top, num_output, kernel_size, pad, stride)
    fp = open(name + '.prototxt', 'w')
    fp.write(conv_layer_str)
    fp.close()



def generate_deconv_layer(name, bottom, top, num_output, kernel_size, pad,stride):
    conv_layer_str = '''layer {
        name: "%s"
        bottom: "%s"
        top: "%s"
        type: "Deconvolution"
        convolution_param {
        num_output: %s
        kernel_size: %s
        pad: %s
        stride: %s
        }
        \n'''%(name, bottom, top, num_output, kernel_size, pad, stride)
    fp = open(name + '.prototxt', 'w')
    fp.write(conv_layer_str)
    fp.close()



def generate_activation_layer(name, bottom, act_type="ReLU"):
    act_layer_str = '''layer {
        name: "%s"
        bottom: "%s"
        top: "%s"
        type: "%s"
        }\n'''%(name, bottom, bottom, act_type)
    fp = open(name + '.prototxt', 'w')
    fp.write(act_layer_str)
    fp.close()



def generate_bn_layer(bn_name, scale_name, bottom):
    bn_layer_str = '''layer {
        name: "%s"
        bottom: "%s"
        top: "%s"
        type: "BatchNorm"
        }
        layer {
        name: "%s"
        bottom: "%s"
        top: "%s"
        type: "Scale"
        scale_param {
        bias_term: true
        }
        \n'''%(bn_name, bottom, bottom, scale_name, bottom, bottom)
    fp = open(scale_name + '.prototxt', 'w')
    fp.write(bn_layer_str)
    fp.close()


def generate_eltwise_layer(name, bottom0, bottom1, top):
    eltwise_layer_str = '''
        layer {
        name: "%s"
        bottom: "%s"
        bottom: "%s"
        top: "%s"
        type: "Eltwise"
        }\n'''%(name, bottom0, bottom1, top)
    fp = open(name + '.prototxt', 'w')
    fp.write(eltwise_layer_str)
    fp.close()


def generate_test():
    f = open("mynet.txt")
    line = f.readline().strip()
    group = line.split(",")
    group = remove_blank(group)
    while line:
        if group[0][0:4] == 'Conv':
            generate_conv_layer(group[0], group[1], group[2], group[3], group[4], group[5], group[6])
            line = f.readline().strip()
            group = line.split(",")
            group = remove_blank(group)
            generate_bn_layer(group[0], group[0], group[1])
            generate_activation_layer(group[0][:-9] + 'relu', group[0])
        elif group[0][0:8] == 'resBlock':
            if group[1].strip() == 'True':
                generate_conv_layer(group[0], group[2], group[3], group[4], group[5], group[6], group[7])
                for i in range(3):
                    line = f.readline().strip()
                    group = line.split(",")
                    group = remove_blank(group)
                    generate_conv_layer(group[0], group[1], group[2], group[3], group[4], group[5], group[6])
                    if i != 2:
                        line = f.readline().strip()
                        group = line.split(",")
                        group = remove_blank(group)
                        generate_bn_layer(group[0], group[0], group[1])
                        generate_activation_layer(group[0][:-9]+'relu', group[0])
            elif group[1].strip() == 'false':
                for i in range(3):
                    if i == 0:
                        generate_conv_layer(group[0], group[2], group[3], group[4], group[5], group[6], group[7])
                        line = f.readline().strip()
                        group = line.split(",")
                        group = remove_blank(group)
                        generate_bn_layer(group[0], group[0], group[1])
                        generate_activation_layer(group[0][:-9]+'relu', group[0])
                    elif i == 1:
                        generate_conv_layer(group[0], group[1], group[2], group[3], group[4], group[5], group[6])
                        line = f.readline().strip()
                        group = line.split(",")
                        group = remove_blank(group)
                        generate_bn_layer(group[0], group[0], group[1])
                        generate_activation_layer(group[0][:-9]+'relu', group[0])
                    elif i == 2:
                        generate_conv_layer(group[0], group[1], group[2], group[3], group[4], group[5], group[6])
                    if i !=2:
                        line = f.readline().strip()
                        group = line.split(",")
                        group = remove_blank(group)
            line = f.readline().strip()
            group = line.split(",")
            group = remove_blank(group)
            generate_eltwise_layer(group[0], group[1], group[2], group[3])
            line = f.readline().strip()
            group = line.split(",")
            group = remove_blank(group)
            generate_bn_layer(group[0], group[0], group[1])
            generate_activation_layer(group[0][:-9] + 'relu', group[0])
        elif group[0][0:16] == 'Conv2d_transpose':
            generate_deconv_layer(group[0], group[1], group[2], group[3], group[4], group[5], group[6])
            line = f.readline().strip()
            group = line.split(",")
            group = remove_blank(group)
            generate_bn_layer(group[0], group[0], group[1])
            generate_activation_layer(group[0][:-9] + 'relu', group[0])
        line = f.readline().strip()
        group = line.split(",")
        group = remove_blank(group)
    f.close()

def remove_blank(group):
    for i in range(len(group)):
        group[i]=group[i].strip()
    return group

def main():
    generate_test()

if __name__ == '__main__':
    main()
