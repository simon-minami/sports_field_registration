'''

given input image, gets the predicted homography
RIGHT NOW: trying to implement on the swim model, swim image etc.
AFTER: once we confirms it works on the swim stuff, we can try training basketball model and then doing inference

TODO: add image that draw court lines on original size image
outputs a variety of image transformations based on the predicted homography
'''

import torch
import copy
from torch import load, unsqueeze, stack, no_grad
from skimage import io

from torchvision import transforms
from torchvision.transforms.functional import rotate as rotate_tensor
import cv2
import os
import matplotlib.pyplot as plt

import numpy as np

from model import deeper_Unet_like, vanilla_Unet
from model_deconv import vanilla_Unet2

from utils.blobs_utils import get_boxes, a_link_to_the_past, get_local_maxima
from video_display_dataloader import get_video_dataloaders
from utils.grid_utils import get_landmarks_positions, get_faster_landmarks_positions, \
    get_homography_from_points, conflicts_managements, display_on_image


def tensor_to_image(out, inv_trans=True, batched=False, to_uint8=True):
    if batched:
        index_shift = 1
    else:
        index_shift = 0
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans:
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    if to_uint8:
        out *= 256
        out = out.astype(np.uint8)
    out = np.swapaxes(out, index_shift + 0, index_shift + 2)
    out = np.swapaxes(out, index_shift + 0, index_shift + 1)
    return out


if __name__ == '__main__':
    torch.cuda.empty_cache()
    size = (256, 256)

    threshold = 0.75

    epochs = 100

    save_projection = True
    # save_projection = False

    field_length = 94
    markers_x = np.linspace(0, field_length, 15)
    field_width = 50
    lines_y = np.linspace(0, field_width, 7)

    model = vanilla_Unet2(final_depth=len(markers_x) + len(lines_y))

    batch_size = 64

    model_path = 'models/bball_unetv2_epoch45.pth'

    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()

    # load and preprocess image following steps in video_display_dataloader.py
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    # img_path = 'dataset/ncaa_bball/images/20230220_WVU_OklahomaSt/frame_2701.jpg'
    img_path = 'images/unseen_test2.png'

    img = io.imread(img_path)
    # img = self.zoom_out(img)
    resized_img = cv2.resize(img, size)

    tensor_img = img_transform(resized_img)
    print(f'shape before .view {tensor_img.size()}')
    tensor_img = tensor_img.view(3, tensor_img.shape[-2], tensor_img.shape[-1])  #
    print(f'shape after .view {tensor_img.size()}')

    tensor_img = tensor_img.unsqueeze(0).cuda()  # add batch dimension and send to gpu
    print(f'shape after adding batch dim {tensor_img.size()}')

    with no_grad():
        batch_out = model(tensor_img)
        print(f'model output before {batch_out.size()}')
        batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)
        print(f'model output after converting back to image {batch_out.shape}')

        #  we don't need the batch dimension in batch_out when we pass into get_faster_landmarks
        resized_img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(resized_img, batch_out[0], threshold,
                                                                          write_on_image=False,
                                                                          lines_nb=len(lines_y),
                                                                          markers_x=markers_x, lines_y=lines_y)

        src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
        H = get_homography_from_points(src_pts, dst_pts, size,
                                       field_length=field_length, field_width=field_width)
        warped_img = cv2.warpPerspective(resized_img, H.astype(float), size)


        draw_img = copy.copy(resized_img)
        # # testing drawing outline
        # # NOTE: drawing the original video key points on original image
        for i, pt in enumerate(src_pts):
            cv2.circle(resized_img, pt, 3, (0, 0, 255), -1)
            cv2.putText(resized_img, f'pt{i}', pt, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

        ##NOTE: corresponding dst points on court grid
        # Plot the points
        x_coords = [coord[0] for coord in dst_pts]
        y_coords = [coord[1] for coord in dst_pts]
        plt.scatter(x_coords, y_coords)

        # Optionally, add labels for each point
        for i, (x, y) in enumerate(dst_pts):
            plt.annotate(f'pt{i}', (x, y))
        # Invert the y-axis
        plt.gca().invert_yaxis()
        # Set the same spacing for x and y axes
        plt.axis('equal')
        plt.savefig('images/grid_out.png')

        # NOTE: trying to go from video to court
        H_video_to_court, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC)
        # cc_top = np.array([
        #     (42, 110)
        # ])
        # print(f'cc_top before: {cc_top}')
        # cv2.circle(img, cc_top[0], 3, (0, 0, 255), -1)
        # cv2.putText(img, 'cc_top', cc_top[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # cc_top_transformed = cv2.perspectiveTransform(cc_top.reshape(-1, 1, 2).astype(float), H_video_to_court.astype(float))
        # cc_top_transformed = cc_top_transformed.reshape(-1, 2)
        # print(f'cc_top after: {cc_top_transformed}')
        # #add to grid
        # plt.scatter(int(cc_top_transformed[0][0]), int(cc_top_transformed[0][1]))
        # plt.annotate(f'cc_top', (int(cc_top_transformed[0][0]), int(cc_top_transformed[0][1])))


        #NOTE: trying to go court to video
        H_court_to_video = np.linalg.inv(H_video_to_court).astype(float)
        rk_top_left = np.array([
            (75, 31)
        ])
        print(f'rk top left before: {rk_top_left}')
        # add to grid
        plt.scatter(rk_top_left[0][0], rk_top_left[0][1])
        plt.annotate(f'rk top left', (rk_top_left[0][0], rk_top_left[0][1]))

        rk_top_left_transformed = cv2.perspectiveTransform(rk_top_left.reshape(-1, 1, 2).astype(float),
                                                      H_court_to_video.astype(float))
        rk_top_left_transformed = rk_top_left_transformed.reshape(-1, 2)
        rk_top_left_transformed = (int(rk_top_left_transformed[0][0]), int(rk_top_left_transformed[0][1]))
        cv2.circle(resized_img, rk_top_left_transformed, 3, (0, 0, 255), -1)
        cv2.putText(resized_img, 'rk top left', rk_top_left_transformed, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f'rk top left after: {rk_top_left_transformed}')

        plt.savefig('images/grid_out.png')







        # # NOTE: trying to draw court lines!!!!
        court_corners = np.array([
            [0, 0], [94, 0], [94, 50], [0, 50]
        ], dtype=float)
        right_key = np.array([
            (75, 19), (94, 19), (94, 31), (75, 31)
        ], dtype=float)
        half_court = np.array([
            (47, 0), (47, 50)
        ], dtype=float)

        court_corners = court_corners.reshape(-1, 1, 2)  # need to reshape for transformation
        right_key = right_key.reshape(-1, 1, 2)
        half_court = half_court.reshape(-1, 1, 2)

        court_corners_video = cv2.perspectiveTransform(court_corners, H_court_to_video)
        right_key_video = cv2.perspectiveTransform(right_key, H_court_to_video)
        half_court_video = cv2.perspectiveTransform(half_court, H_court_to_video)
        # print(court_corners_video, court_corners_video.shape)

        court_corners_video = court_corners_video.astype(int).reshape(-1, 2)
        right_key_video = right_key_video.astype(int).reshape(-1, 2)
        half_court_video = half_court_video.astype(int).reshape(-1, 2)

        # prinkt(court_corners_video, court_corners_video.shape)
        pt1 = court_corners_video[0, :]
        pt2 = court_corners_video[1, :]
        pt3 = court_corners_video[2, :]
        pt4 = court_corners_video[3, :]

        cv2.line(draw_img, pt1, pt2, (0, 0, 255), 3)
        cv2.line(draw_img, pt2, pt3, (0, 0, 255), 3)
        cv2.line(draw_img, pt3, pt4, (0, 0, 255), 3)
        cv2.line(draw_img, pt4, pt1, (0, 0, 255), 3)

        pt1 = right_key_video[0, :]
        pt2 = right_key_video[1, :]
        pt3 = right_key_video[2, :]
        pt4 = right_key_video[3, :]

        cv2.line(draw_img, pt1, pt2, (0, 0, 255), 3)
        cv2.line(draw_img, pt2, pt3, (0, 0, 255), 3)
        cv2.line(draw_img, pt3, pt4, (0, 0, 255), 3)
        cv2.line(draw_img, pt4, pt1, (0, 0, 255), 3)

        pt1 = half_court_video[0, :]
        pt2 = half_court_video[1, :]

        cv2.line(draw_img, pt1, pt2, (0, 0, 255), 3)



        #NOTE: trying to draw court lines on original scale image
        # # NOTE: trying to draw court lines!!!!
        court_corners = np.array([
            [0, 0], [94, 0], [94, 50], [0, 50]
        ], dtype=float)
        right_key = np.array([
            (75, 19), (94, 19), (94, 31), (75, 31)
        ], dtype=float)
        half_court = np.array([
            (47, 0), (47, 50)
        ], dtype=float)

        court_corners = court_corners.reshape(-1, 1, 2)  # need to reshape for transformation
        right_key = right_key.reshape(-1, 1, 2)
        half_court = half_court.reshape(-1, 1, 2)

        print(f'original shape: {img.shape}')
        scale_factor = np.eye(3)
        scale_factor[0, 0] = img.shape[1] / size[0]
        scale_factor[1, 1] = img.shape[0] / size[1]
        H_court_to_video_scaled = np.matmul(scale_factor, H_court_to_video)

        court_corners_video = cv2.perspectiveTransform(court_corners, H_court_to_video_scaled)
        right_key_video = cv2.perspectiveTransform(right_key, H_court_to_video_scaled)
        half_court_video = cv2.perspectiveTransform(half_court, H_court_to_video_scaled)
        # print(court_corners_video, court_corners_video.shape)

        court_corners_video = court_corners_video.astype(int).reshape(-1, 2)
        right_key_video = right_key_video.astype(int).reshape(-1, 2)
        half_court_video = half_court_video.astype(int).reshape(-1, 2)

        # prinkt(court_corners_video, court_corners_video.shape)
        pt1 = court_corners_video[0, :]
        pt2 = court_corners_video[1, :]
        pt3 = court_corners_video[2, :]
        pt4 = court_corners_video[3, :]

        cv2.line(img, pt1, pt2, (0, 0, 255), 3)
        cv2.line(img, pt2, pt3, (0, 0, 255), 3)
        cv2.line(img, pt3, pt4, (0, 0, 255), 3)
        cv2.line(img, pt4, pt1, (0, 0, 255), 3)

        pt1 = right_key_video[0, :]
        pt2 = right_key_video[1, :]
        pt3 = right_key_video[2, :]
        pt4 = right_key_video[3, :]

        cv2.line(img, pt1, pt2, (0, 0, 255), 3)
        cv2.line(img, pt2, pt3, (0, 0, 255), 3)
        cv2.line(img, pt3, pt4, (0, 0, 255), 3)
        cv2.line(img, pt4, pt1, (0, 0, 255), 3)

        pt1 = half_court_video[0, :]
        pt2 = half_court_video[1, :]

        cv2.line(img, pt1, pt2, (0, 0, 255), 3)
        print(f'there are {len(src_pts)} src pts: {src_pts}')

        print(f'there are {len(dst_pts)} dst pts: {dst_pts}')

        print(f'homography M: {H}')
        cv2.imwrite('images/test_output.jpg', resized_img)
        cv2.imwrite('images/warp_output.jpg', warped_img)
        cv2.imwrite('images/output_lines.jpg', draw_img)
        cv2.imwrite('images/output_lines_original.jpg', img)

