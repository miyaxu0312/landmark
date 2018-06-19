import argparse
import os
from glob import glob
import ast
import numpy as np
from skimage.io import imread, imsave
from api import PRN
from skimage.transform import rescale, resize
from generate_pos import  get_pos


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib=args.isDlib)
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)

    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, _] = image.shape
        if args.isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000./max_size)
                image = (image*255).astype(np.uint8)
            cropped_image, tform, resolution_np = prn.process(image)                 # use dlib to detect face
        else:                                                  #without dlib should set the parameters of imageinfo
            if image.shape[1] == image.shape[2]:
                cropped_image, tform, resolution_np= resize(image, (256, 256))
                #pos = prn.net_forward(image/255.)             # input image has been cropped to 256x256,and set to be 0-1 values
            else:
                box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                cropped_image, tform, resolution_np  = prn.process(image, box)
        imsave(os.path.join(save_folder, name + '_pre_processed.jpg'), image)
        np.savetxt(os.path.join(save_folder,'tform.txt'), tform)
        #network test to get the output images for further process
        #get_pos(cropped_image, tform, resolution_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Landmark pre_processing')
    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/pre_processed', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')

    main(parser.parse_args())
