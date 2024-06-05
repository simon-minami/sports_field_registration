'''
Use to check whether dataset created correctly
'''


import os
import os.path as osp
import cv2
import numpy as np

root = osp.join('dataset', 'ncaa_bball')
# image_path = osp.join(root, 'images', '20230220_WVU_OklahomaSt')
# annotation_path = osp.join(root, 'annotations', '20230220_WVU_OklahomaSt')
image_path = osp.join(root, 'images', '20240106_duke_notredame_h1')
annotation_path = osp.join(root, 'annotations', '20240106_duke_notredame_h1')

frames = sorted(os.listdir(image_path))
homographies = sorted(os.listdir(annotation_path))


for frame, homo in zip(frames, homographies):

    img = cv2.imread(osp.join(image_path, frame))
    print(img.shape)

    H_video_to_court = np.load(osp.join(annotation_path, homo))
    warped_img = cv2.warpPerspective(img, H_video_to_court.astype(float), (1280, 720))
    print(f'warped: {warped_img.shape}')
    print(f'showing: {frame}')


    cv2.namedWindow('warped', cv2.WINDOW_NORMAL)
    cv2.imshow('warped', warped_img)
    if cv2.waitKey(0) == ord('s'):
        cv2.imwrite("C:/Users/simon/Downloads/homogpic.png", img)

cv2.destroyAllWindows()

