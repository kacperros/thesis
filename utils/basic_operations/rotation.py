import numpy as np
import cv2


def center_rotate(src, angle, scale=1.):
    if angle == 0:
        return src
    h, w, channels = src.shape
    rad_angle = np.deg2rad(angle)
    new_w = (abs(np.sin(rad_angle) * h) + abs(np.cos(rad_angle) * w)) * scale
    new_h = (abs(np.cos(rad_angle) * h) + abs(np.sin(rad_angle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((new_w * 0.5, new_h * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_w - w) * 0.5, (new_h - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(np.ceil(new_w)), int(np.ceil(new_h))), flags=cv2.INTER_LANCZOS4)
