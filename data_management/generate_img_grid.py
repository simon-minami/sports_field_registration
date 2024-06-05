'''

updated version of homo_to_points.py that can handle multiple game foldres
NOTES:
    got rid of the display function to make it simpler
'''
import pickle
import pandas as pd
import numpy as np
from random import random
import os
import cv2
from cv2 import warpPerspective, findHomography, GaussianBlur, resize, imread
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



def get_matrix(data, out_size=(256, 256), in_size=(1000, 500)) :
    src_pts = data.source_points.numpy()
    src_pts[:, 0] = src_pts[:, 0] * out_size[0] / in_size[0]
    src_pts[:, 1] = src_pts[:, 1] * out_size / in_size[1]
    dst_pts = data.destination_points.numpy()
    dst_pts[:, 0] = dst_pts[:, 0] * out_size[0] / in_size[0]
    dst_pts[:, 1] = dst_pts[:, 1] * out_size[1] / in_size[1] - out_size[1]

    matrix, _ = findHomography(src_pts, dst_pts)
    return matrix


def main():
    # save_path = "./dense_grid/"
    # template_path = './dense_grid.npy'
    # save_path = "grids"
    save_path = "../dataset/ncaa_bball/grids"
    matrices_path = '../dataset/ncaa_bball/annotations'
    template_path = 'grid.npy'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    files_list = os.listdir(save_path)

    # TODO: we'll have to change this around to handle multple game folders
    labels_path = '../dataset/ncaa_bball/annotations/20230220_WVU_OklahomaSt'
    # files = os.listdir(labels_path)
    # files = [f for f in files if f.endswith(".npy")]  # npy file names
    # matrices = [np.load(os.path.join(labels_path, f)) for f in files]  # actual matrices loaded

    out_size = (1280, 720)
    final_size = (256, 256)

    template = np.load(template_path)
    print(f'template before swap: {template.shape}')

    template = np.swapaxes(template, 2, 0)
    template = np.swapaxes(template, 0, 1)
    # template = resize(template, out_size)
    print(f'template after swap: {template.shape}')
    train_file = '../dataset/ncaa_bball/train.txt'
    # train file simply contains names of each game we want to use like 20230217_washingtonst_oregon etc.

    with open(train_file, 'r') as file:
        # Iterate through each line in the train file
        for line in file:
            game_folder = line.strip()  # need line.strip() to get rid of \n
            file_names = os.listdir(os.path.join(matrices_path, game_folder))  # should have all the npy file names
            print(file_names)
            matrices = [np.load(os.path.join(matrices_path, game_folder, f)) for f in file_names]  # actual matrices loaded
            # print(matrices)

            game_save_path = os.path.join(save_path, game_folder)
            if not os.path.exists(game_save_path):
                os.mkdir(game_save_path)

            for file_name, h in zip(file_names, matrices):  # generate grid for each img

                heatmap_path = os.path.join(game_save_path, file_name)
                # h = h[0].numpy()

                if not np.isnan(h).any():
                    # scale_factor = np.eye(3)  # scale factor was used on soccer dataset which had different format.
                    # # scale_factor[0, 0] = out_size[0] / 115
                    # # scale_factor[1, 1] = out_size[1] / 74
                    # scale_factor[0, 0] = out_size[0] / 94
                    # scale_factor[1, 1] = out_size[1] / 50
                    # h = scale_factor @ h

                    # h maps from video to 2D court, so we inverse to get 2D court -> video
                    h_back = np.linalg.inv(h)

                    result = warpPerspective(template, h_back, out_size)
                    print(f'result: {result.shape}')
                    result = resize(result, final_size)
                    print(f'result after final resize: {result.shape}')
                    # cv2.waitKey(0)

                    result = GaussianBlur(result, (5, 5), 0)
                    result[result != 0] = 255
                    # result = GaussianBlur(result, (3, 3), 0)
                    result = result.astype(np.uint8)

                    np.save(heatmap_path, result)
if __name__ == "__main__" :
    main()


