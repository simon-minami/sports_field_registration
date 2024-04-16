'''
contains homography functions that will be used in main tracking data loop (tracking_demo.py)
'''
import cv2
import numpy as np
import torch
from torchvision import transforms
from sports_field_registration.utils.grid_utils import get_faster_landmarks_positions, conflicts_managements

def getHomographyMatrix(model, img, video_size, size=(256, 256), field_length=94, field_width=50, threshold=0.75):
    '''
    model: should be loaded and in eval mode
    video_size: input video dimensions (w, h)
    size: constant image resize used by homography model (don't change)
    '''
    markers_x = np.linspace(0, field_length, 15)
    lines_y = np.linspace(0, field_width, 7)
    width = video_size[0]
    height = video_size[1]

    resized_img, tensor_img = preprocess(img, size)
    batch_out = model(tensor_img)
    # print(f'model output before {batch_out.size()}')
    batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)
    # print(f'model output after converting back to image {batch_out.shape}')

    #  we don't need the batch dimension in batch_out when we pass into get_faster_landmarks
    resized_img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(resized_img, batch_out[0], threshold,
                                                                              write_on_image=False,
                                                                              lines_nb=len(lines_y),
                                                                              markers_x=markers_x, lines_y=lines_y)


    src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
    if len(src_pts) < 4:
        print('homo could not be caluclated')
        return None

    H_video_to_court, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, ransacReprojThreshold=3)

    #NOTE: H_video_to_court maps from 256x256 video to court diagram \
    # we want to it to go from video_size[0] x video_size[1] to court diagram

    scale_factor = np.eye(3)
    scale_factor[0, 0] = size[0] / width
    scale_factor[1, 1] = size[1] / height
    H_video_to_court_scaled = np.matmul(scale_factor, H_video_to_court)
    return H_video_to_court_scaled




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
    #NOTE: cv2 default reads in BGR format, need to convert to RGB for model to work (idk really why, just it that way)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized_img = cv2.resize(rgb_frame, size)

    tensor_img = img_transform(resized_img)
    # print(f'shape before .view {tensor_img.size()}')
    tensor_img = tensor_img.view(3, tensor_img.shape[-2], tensor_img.shape[-1])  #
    # print(f'shape after .view {tensor_img.size()}')

    tensor_img = tensor_img.unsqueeze(0).cuda()  # add batch dimension and send to gpu
    # print(f'shape after adding batch dim {tensor_img.size()}')
    return resized_img, tensor_img

def tensor_to_image(out, inv_trans=True, batched=False, to_uint8=True) :
    if batched : index_shift = 1
    else : index_shift = 0
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans :
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    if to_uint8 :
        out *= 256
        out = out.astype(np.uint8)
    out = np.swapaxes(out, index_shift + 0, index_shift + 2)
    out = np.swapaxes(out, index_shift + 0, index_shift + 1)
    return out


def getTrackingData(homo_matrix, tlwhs, ball_bbox, obj_ids=None, frame_id=0):
    '''
    applys homographic transformation and returns formated tracking data for given frame
    Args:
        calib: Calib object returned by estimate calib
        tlwhs: player bounding boxes in form list of [x1, y1, w, h]
        ball_bbox: ball bounding boxes in form [xmin, ymin, xmax, ymax, conf, class id]
        obj_ids: player ids
        frame_id:

    Returns: array of new tracking data

    '''
    # take tlwhs which is list of BBs in form [x1, y1, w, h] and turn it to player coordinate
    # should have shape (num tracks, 2)
    num_tracks = len(tlwhs)

    # create player id column
    player_ids = np.array(obj_ids).reshape(num_tracks, 1)  # Example: Column of zeros

    video_coords = np.array(list(map(lambda bb: (bb[0] + bb[2] / 2, bb[1] + bb[3]), tlwhs)))
    # do same thing for ball coordinate, which is in different format
    if ball_bbox is not None: # will be none if number of detections is 0 or best detection didn't meet threshold
        ball_coords = np.array([(ball_bbox[0] + (ball_bbox[2] - ball_bbox[0]) / 2, ball_bbox[3])])
        video_coords = np.vstack((video_coords, ball_coords))
        num_tracks += 1
        player_ids = np.vstack([player_ids, [-1]])  # update player id with -1 for the ball

    # create frame id column
    frame_ids = np.full((num_tracks, 1), frame_id)  # Example: Column of ones




    pts = video_coords.reshape(-1, 1, 2)  # need to reshape for transformation
    # print(f'dubug: number of tracks: {num_tracks}, homo_matrix: {homo_matrix}')
    transformed_coords = cv2.perspectiveTransform(pts, homo_matrix).reshape(num_tracks, 2)



    # Concatenate the new columns with the original array
    tracking_data = np.hstack((frame_ids, player_ids, transformed_coords))
    # print(f'tracking_data: {tracking_data.shape}')
    return tracking_data