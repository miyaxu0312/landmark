import numpy as np


def get_pos(cropped_pos,tform,resolution_op):
    cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, [resolution_op, resolution_op, 3])

    return pos