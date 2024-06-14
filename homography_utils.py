'''
contains homography functions that will be used in main tracking data loop (tracking_demo.py)
'''
import cv2
import numpy as np
import torch
from torchvision import transforms
import sys

for path in sys.path:
    print(path)
from sports_field_registration.utils.grid_utils import get_faster_landmarks_positions, conflicts_managements


def get_homography_matrix(model, img, video_size, size=(256, 256), field_length=94, field_width=50, threshold=0.75):
    '''
    Given image of court, predicts homogrpahy matrix mapping from video to 2D court space
    Args:
        model (vanilla_Unet2_: should be loaded and in eval mode (should be vanilla_Unet2)
        video_size (tuple): input video dimensions (w, h)
        size (tuple): constant image resize used by homography model (w, h) (don't change)
    Returns:
         numpy matrix: the video to court homography matrix
    '''
    # using 15x7 uniform grid representation of court
    markers_x = np.linspace(0, field_length, 15)
    lines_y = np.linspace(0, field_width, 7)
    width = video_size[0]
    height = video_size[1]

    with torch.no_grad():
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

    H_video_to_court, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, ransacReprojThreshold=3)

    # H_video_to_court maps from 256x256 video to court diagram
    # we want to it to go from video_size[0] x video_size[1] to court diagram
    # current code works, but its redundant (TODO: should be a way to do this without all the inverting)
    H_court_to_video = np.linalg.inv(H_video_to_court)
    scale_factor = np.eye(3)
    scale_factor[0, 0] = width / size[0]
    scale_factor[1, 1] = height / size[0]
    H_court_to_video_scaled = np.matmul(scale_factor, H_court_to_video)
    # now H_court_to_video maps from court to video width x height
    # invert to get width x height to court
    return np.linalg.inv(H_court_to_video_scaled)


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
