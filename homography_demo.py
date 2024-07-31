'''
input: game video
output: demo video with predicted homography lines drawn

cmd line arguments
--input_path
--model_path
--output_path
'''
import argparse
import cv2
import numpy as np
from model_deconv import vanilla_Unet2
import torch
from torchvision import transforms
from utils.grid_utils import get_faster_landmarks_positions, conflicts_managements, get_homography_from_points
from homography_utils import get_homography_matrix


def make_parser():
    parser = argparse.ArgumentParser("Jacquelin et al Homography Demo")
    parser.add_argument("--input_path", default=None, type=str, help="input video path")
    parser.add_argument("--model_path", default=None, type=str, help="homography model path (.pth file)")
    parser.add_argument("--output_path", default=None, type=str, help="output video path")
    return parser



def draw_court_lines(frame, H_court_to_video):
    '''
    draw court lines on given frame
    '''
    court_corners = np.array([
        [0, 0], [94, 0], [94, 50], [0, 50]
    ], dtype=float)
    half_court = np.array([
        (47, 0), (47, 50)
    ], dtype=float)

    court_corners = court_corners.reshape(-1, 1, 2)  # need to reshape for transformation
    half_court = half_court.reshape(-1, 1, 2)

    court_corners_video = cv2.perspectiveTransform(court_corners, H_court_to_video)
    half_court_video = cv2.perspectiveTransform(half_court, H_court_to_video)

    court_corners_video = court_corners_video.astype(int).reshape(-1, 2)
    half_court_video = half_court_video.astype(int).reshape(-1, 2)

    pt1 = court_corners_video[0, :]
    pt2 = court_corners_video[1, :]
    pt3 = court_corners_video[2, :]
    pt4 = court_corners_video[3, :]

    cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
    cv2.line(frame, pt2, pt3, (0, 0, 255), 3)
    cv2.line(frame, pt3, pt4, (0, 0, 255), 3)
    cv2.line(frame, pt4, pt1, (0, 0, 255), 3)

    pt1 = half_court_video[0, :]
    pt2 = half_court_video[1, :]

    cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
    return frame


def main(args):
    torch.cuda.empty_cache()
    # initialize video reader and writer
    cap = cv2.VideoCapture(args.input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'input video: {width, height}')

    vid_writer = cv2.VideoWriter(
        args.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # load homography model and define some constants
    field_length = 94
    markers_x = np.linspace(0, field_length, 15)
    field_width = 50
    lines_y = np.linspace(0, field_width, 7)

    model = vanilla_Unet2(final_depth=len(markers_x) + len(lines_y))
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    model.eval()

    # with torch.no_grad():
    ret_val, frame = cap.read()

    while ret_val:
        # predict homography
        H_video_to_court = get_homography_matrix(model, frame, src_dims=(width, height))
        H_court_to_video = np.linalg.inv(H_video_to_court)

        # draw predicted court lines on frame
        frame = draw_court_lines(frame, H_court_to_video)

        # write to video
        vid_writer.write(frame)

        # get current frame
        frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_id % 120 == 0:
            print(f'processed {frame_id} frames')
            cv2.imwrite(f'images/homography_frame{frame_id}.jpg', frame)

        ret_val, frame = cap.read()


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
