import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
from api import PRN

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize


def main(args):
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    resolution = args.resolution
    face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32)  # get valid vertices in the pos map
    uv_kpt_ind = np.loadtxt('./Data/uv-data/uv_kpt_ind.txt').astype(np.int32)  # 2 x 68 get kpt
    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('/')[-1][:-4]

        # read image
        image = imread(image_path)
        [h, w, _] = image.shape
        tform = np.loadtxt(os.path.join(save_folder,'tform.txt'), delimiter=',')
        pos = get_pos(image, tform, resolution)

        if args.isPose:
            # 3D vertices
            vertices = get_vertices(pos,resolution, face_ind)      # get valid points
            if args.isFront:
                save_vertices = frontalize(vertices)   # affine matrix to frontalize
            else:
                save_vertices = vertices.copy()     # 68 * 3
            save_vertices[:,1] = h - 1 - save_vertices[:,1]      # control the scope

        if args.isKpt or args.isShow:
            # get landmarks
            kpt = get_landmarks(pos, uv_kpt_ind)
            np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

        if args.isPose:  # camera matrix
            # estimate pose
            camera_matrix, pose = estimate_pose(vertices)
            np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose)
            np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix)



def get_pos(cropped_pos,tform,resolution_op):
    cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / tform[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, [resolution_op, resolution_op, 3])

    return pos


def get_vertices(pos,resolution_op,face_ind):     # get the valid points of 3d face
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [resolution_op**2, -1])
    vertices = all_vertices[face_ind, :]     #43867

    return vertices


def get_landmarks(pos, uv_kpt_ind):   #get the kpt in the test images according to the x-y axis of the standard uv file
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    '''
    kpt = pos[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
    return kpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Landmark post_processing')
    parser.add_argument('-i', '--inputDir', default='TestImages/networkOutput', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/post_processed', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--isKpt', default=True, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=True, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isImage', default=True, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=True, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    parser.add_argument('--resolution', default=256, type=ast.literal_eval,
                        help='the resolution of the images')
    main(parser.parse_args())
