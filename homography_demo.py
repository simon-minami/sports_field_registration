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
from utils.grid_utils import get_faster_landmarks_positions, conflicts_managements


def make_parser():
    parser = argparse.ArgumentParser("Jacquelin et al Homography Demo")
    parser.add_argument("--input_path", default=None, type=str, help="input video path")
    parser.add_argument("--model_path", default=None, type=str, help="homography model path (.pth file)")
    parser.add_argument("--output_path", default=None, type=str, help="output video path")
    return parser


def preprocess(img, size):
    '''
    apply necessary preprocess to img before inputing into model
    '''
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    img = cv2.resize(img, size)

    tensor_img = img_transform(img)
    print(f'shape before .view {tensor_img.size()}')
    tensor_img = tensor_img.view(3, tensor_img.shape[-2], tensor_img.shape[-1])  #
    print(f'shape after .view {tensor_img.size()}')

    tensor_img = tensor_img.unsqueeze(0).cuda()  # add batch dimension and send to gpu
    print(f'shape after adding batch dim {tensor_img.size()}')
    return tensor_img

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

def draw_court_lines(frame, H_court_to_video):
    '''
    draw court lines
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
    size = (256, 256)
    threshold = 0.75
    # initialize video reader and writer

    cap = cv2.VideoCapture(args.input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

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

    with torch.no_grad():
        ret_val, frame = cap.read()
        while ret_val:
            # apply necessary preprocessing to frame
            # load and preprocess image following steps in video_display_dataloader.py
            tensor_img = preprocess(frame, size)
            batch_out = model(tensor_img)
            print(f'model output before {batch_out.size()}')
            batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)
            print(f'model output after converting back to image {batch_out.shape}')

            #  we don't need the batch dimension in batch_out when we pass into get_faster_landmarks
            img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(img, batch_out[0], threshold,
                                                                              write_on_image=False,
                                                                              lines_nb=len(lines_y),
                                                                              markers_x=markers_x, lines_y=lines_y)

            src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)

            H_video_to_court, _ = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC)
            H_court_to_video = np.linalg.inv(H_video_to_court).astype(float)

            frame = draw_court_lines(frame, H_court_to_video)

            vid_writer.write(frame)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
