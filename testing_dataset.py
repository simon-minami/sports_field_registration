'''
simple script to check which direction the swim image homographies are
conclusion: matrix is 2D (court) -> video
however, the output isn't showing up right, might have to do with img resizing?

'''


import os
import os.path as osp
import cv2
import numpy as np

if __name__ == '__main__':

    image_path = 'dataset/ncaa_bball/images/20230220_WVU_OklahomaSt/frame_4501.jpg'
    annotation_path = 'dataset/ncaa_bball/annotations/20230220_WVU_OklahomaSt/frame_4501.npy'

    print('test')
    print(np.load(annotation_path))

    field_length = 50
    field_width = 25

    frame = cv2.imread(image_path)
    print(frame.shape)
    # frame = cv2.resize(frame, (256, 256))
    H = np.load(annotation_path).astype(float)

    warped = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
    # H = np.linalg.inv(H)
    # pool_corners = np.array([
    #         [0,0], [field_length, 0], [field_length, field_width], [0, field_width]
    #     ], dtype=float)
    #
    # pool_corners = pool_corners.reshape(-1, 1, 2)
    # pool_corners_video = cv2.perspectiveTransform(pool_corners, H)
    # pool_corners_video = pool_corners_video.astype(int).reshape(-1, 2)
    #
    # pt1 = pool_corners_video[0, :]
    # pt2 = pool_corners_video[1, :]
    # pt3 = pool_corners_video[2, :]
    # pt4 = pool_corners_video[3, :]
    #
    # print(pt1, pt2, pt3, pt4)
    #
    # cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
    # cv2.line(frame, pt2, pt3, (0, 0, 255), 3)
    # cv2.line(frame, pt3, pt4, (0, 0, 255), 3)
    # cv2.line(frame, pt4, pt1, (0, 0, 255), 3)
    cv2.imshow('frame', frame)
    cv2.imwrite('images/initial.jpg', frame)
    cv2.imwrite('images/warped.jpg', warped)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

