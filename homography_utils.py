'''
contains homography functions that will be used in main tracking data loop (tracking_demo.py)
'''
import cv2
import numpy as np
import torch
from torchvision import transforms
from utils.grid_utils import get_faster_landmarks_positions, conflicts_managements
from cv2 import warpPerspective
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon
import os.path as osp

# IMPORTANT NOTE ON READING IMGS:
# Model needs images to be in RGB format to work, but cv2 reads imgs default in BGR format
# Because of this we have preprocessing in place to flip from BGR to RGB
# If you use different img read function like skimage.io.imread (which reads in RGB default)
# the color channels will get flipped to BGR which will really mess up the homography predictions
def get_homography_matrix(model, img, src_dims=(1280, 720), size=(256, 256), dst_dims=(94, 50), threshold=0.75):
    '''
    Given image of court, predicts homography matrix mapping from video to 2D court space
    Scales H to specified src and dst dimensions
    Args:
        model (vanilla_Unet2_: should be loaded and in eval mode (should be vanilla_Unet2)
        video_size (tuple): input video dimensions (w, h)
        size (tuple): constant image input dimensions expected homography model (w, h) (don't change)
        src_dims: width, height of src space you want for outputted H
        dst_dims: width, height of dst space you want for outputted H
    Returns:
         numpy matrix: the video to court homography matrix
    '''
    # using 15x7 uniform grid representation of court
    dst_width, dst_height = dst_dims
    src_width, src_height = src_dims
    markers_x = np.linspace(0, dst_width, 15)
    lines_y = np.linspace(0, dst_height, 7)


    with torch.inference_mode():
        resized_img, tensor_img = preprocess(img, size)
        grid_output = model(tensor_img)
        grid_output = tensor_to_image(grid_output, inv_trans=False, batched=True, to_uint8=False)

    #  we don't need the batch dimension in grid_output when we pass into get_faster_landmarks
    resized_img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(resized_img, grid_output[0], threshold,
                                                                              write_on_image=False,
                                                                              lines_nb=len(lines_y),
                                                                              markers_x=markers_x, lines_y=lines_y)

    src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
    if len(src_pts) < 4:
        print('homo could not be calculated')
        return None
    # right now src is in 256x256, dst is in dst_width x dst_height
    # now we have to scale the src pts to desired dimension
    src_scale_x = src_width / size[0]
    src_scale_y = src_height / size[1]

    src_pts = [(x*src_scale_x, y*src_scale_y) for (x, y) in src_pts]
    H_video_to_court, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, ransacReprojThreshold=3)


    return H_video_to_court


def preprocess(img, size):
    '''
    apply necessary preprocess to img before inputing into model
    also returns resized version of original
    '''
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    # Convert from BGR to RGB
    # cv2 default reads in BGR format, need to convert to RGB for model to work (idk really why, just it that way)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_frame, size)
    tensor_img = img_transform(resized_img)
    tensor_img = tensor_img.view(3, tensor_img.shape[-2], tensor_img.shape[-1])  #
    tensor_img = tensor_img.unsqueeze(0).cuda()  # add batch dimension and send to gpu
    return resized_img, tensor_img


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


def get_tracking_data(homo_matrix, list_top_left_width_height, ball_bbox, obj_ids=None, frame_id=0):
    '''
    applies homographic transformation and returns formatted tracking data for given frame
    Args:
        list_top_left_width_height: player bounding boxes in form list of [x1, y1, w, h]
        ball_bbox: ball bounding box in form [xmin, ymin, xmax, ymax, conf, class id]
        obj_ids: player ids
        frame_id:

    Returns: array of new tracking data

    '''
    # take list_top_left_width_height which is list of BBs in form [x1, y1, w, h] and turn it to player coordinate
    # should have shape (num tracks, 2)
    num_tracks = len(list_top_left_width_height)

    # create player id column
    player_ids = np.array(obj_ids).reshape(num_tracks, 1)  # Example: Column of zeros

    video_coords = np.array(list(map(lambda bb: (bb[0] + bb[2] / 2, bb[1] + bb[3]), list_top_left_width_height)))
    # do same thing for ball coordinate, which is in different format
    if ball_bbox is not None:  # will be none if number of detections is 0 or best detection didn't meet threshold
        ball_coords = np.array([(ball_bbox[0] + (ball_bbox[2] - ball_bbox[0]) / 2, ball_bbox[3])])
        video_coords = np.vstack((video_coords, ball_coords))
        num_tracks += 1
        player_ids = np.vstack([player_ids, [-1]])  # update player id with -1 for the ball

    # create frame id column
    frame_ids = np.full((num_tracks, 1), frame_id)  # Example: Column of ones

    pts = video_coords.reshape(-1, 1, 2)  # need to reshape for transformation
    transformed_coords = cv2.perspectiveTransform(pts, homo_matrix).reshape(num_tracks, 2)

    # Concatenate the new columns with the original array
    tracking_data = np.hstack((frame_ids, player_ids, transformed_coords))
    return tracking_data



#  IOU calculations
# H_path = '/content/drive/MyDrive/repos/sports_field_registration/dataset/ncaa_bball/annotations'
# img_path = '/content/drive/MyDrive/repos/sports_field_registration/dataset/ncaa_bball/images'
def calc_iou_part(img, H_true, H_pred):
    '''
    Given img, H_true, H_pred calculates the iou part
    NEED to keep in mind dimensions
    H_true maps from by default H_true maps from 1280x720 to 1280x720
    img is from original data set so its 1280x720

    '''
    # assume img is 1280x720, H_true maps from 1280x720 to 1280x720
    true_projection = warpPerspective(img, H_true, (1280, 720))
    # np.unique(true_projection[0, :, :])
    pred_projection = warpPerspective(img, H_pred, (1280, 720))

    # projections are original 3 color channels, we want to compress because
    # we only care about where the court was and wasn't projected
    truth_mask = np.where(true_projection.max(axis=2) > 0, 1, 0)
    pred_mask = np.where(pred_projection.max(axis=2) > 0, 1, 0)

    # now, calculate the IOU part
    # boolean comparison creates same size array, true if condition true, false otherwise
    intersection = np.sum((truth_mask == 1) & (pred_mask == 1))
    union = np.sum((truth_mask == 1) | (pred_mask == 1))
    iou_part = intersection / union

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].imshow(true_projection)
    # axs[0].set_title('ground truth projection')
    # axs[1].imshow(pred_projection)
    # axs[1].set_title(f'pred projection, iou: {iou_part}')

    return iou_part


def calc_iou_whole(H_true, H_pred):
    '''
    Given H_true, H_pred calculates the iou whole
    unlike calc_iou_part we don't need img because we're not projecting the img, just the corners
    NEED to keep in mind dimensions
    H_true maps from by default H_true maps from 1280x720 to 1280x720
    img is from original data set so its 1280x720

    '''
    corners_truth = np.array([(0, 0), (1280, 0), (1280, 720), (0, 720)]).reshape(-1, 1, 2).astype(
        float)  # need to reshape for transformation
    corners_in_video = cv2.perspectiveTransform(corners_truth, np.linalg.inv(H_true))

    corners_pred = cv2.perspectiveTransform(corners_in_video, H_pred)

    # now we can calculate the iou whole
    court_truth = Polygon(corners_truth.squeeze())
    court_pred = Polygon(corners_pred.squeeze())
    intersection = shapely.intersection(court_truth, court_pred).area
    union = shapely.union(court_truth, court_pred).area
    iou_whole = intersection / union

    # optional visualization stuff
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # ax.set_xlim([-200, 1480])
    # ax.set_ylim([-200, 920])

    # corners_truth_closed = np.concatenate([corners_truth.squeeze(), corners_truth[0, :]])
    # print(corners_truth.shape, corners_truth_closed.shape)

    # corners_pred_closed = np.concatenate([corners_pred.squeeze(), corners_pred[0, :]])
    # ax.plot(corners_truth_closed[:, 0], corners_truth_closed[:, 1], color='blue')
    # ax.plot(corners_pred_closed[:, 0], corners_pred_closed[:, 1], color='red')
    # ax.set_title(f'blue truth, red pred, iou_whole: {iou_whole}')
    # plt.show()
    return iou_whole


def get_iou_part_and_whole(model, test_dataloader, img_path, H_path):
    '''
    gets the iou part and whole on test dataset
    iou_part:
    1. get truth and pred projections
    2. create truth and pred binary masks
    3. do matrix summing etc to calc iou

    iou_whole:
    1. get ground truth corners (assume 1280x720)
    2. use H_true inv to get ground truth video corners
    3. use H_pred to get predicted corners
    4. use Polygons to find intersection, union
    '''
    iou_part = []
    iou_whole = []

    for batch in test_dataloader:
        for H_true_name in batch['H_name']:
            # H_true_name is game/frame.npy ex: 20230217_washingtonst_oregon/frame_541.npy
            # shouldn't need a bgr to rgb convert because test dataloader loaded images using skimage.io.imread NOT cv2
            # confusing bruh
            # once we finish this run evaluation comparing with conversion to without just to see what happens
            img = cv2.imread(osp.join(img_path, H_true_name.replace('npy', 'jpg')))
            H_true = np.load(osp.join(H_path, H_true_name))
            H_pred = get_homography_matrix(model, img, src_dims=(1280, 720), dst_dims=(1280, 720))
            # print(H_true_name)
            # img is 1280x720, H_true maps from 1280x720 to 1280x720

            iou_part.append(calc_iou_part(img, H_true, H_pred))
            iou_whole.append(calc_iou_whole(H_true, H_pred))
    return iou_part, iou_whole


