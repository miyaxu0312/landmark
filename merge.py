# -*- coding:utf-8*-
import sys
import os
import os.path

def MergeTxt(filename,netpath,parapath,outpath,outname):
    #k = open(para_path+filename, 'a+')
    for file in os.listdir(para_path):
        if(file == filename):
            result = open(os.path.join(outpath, outname), 'a+')
            net_data = open(os.path.join(net_path, filename))
            para_data = open(os.path.join(parapath, file))
            result.write("\n")
            result.write(net_data.read()+"\n")
            result.write(para_data.read()+"\n")
            result.write("}")

k.close()
print "finished"


if __name__ == '__main__':
    filepath="./"
    net_path = "./net"
    para_path = "./para"
    save_path = "./added_para"
    for file in os.listdir(net_path):
        outname = file
        MergeTxt(file,net_path,para_path,outname)

