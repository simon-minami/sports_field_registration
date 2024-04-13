'''

given input image, gets the predicted homography
RIGHT NOW: trying to implement on the swim model, swim image etc.
AFTER: once we confirms it works on the swim stuff, we can try training basketball model and then doing inference
'''
import torch
from torch import load, unsqueeze, stack, no_grad
from skimage import io

from torchvision import transforms
from torchvision.transforms.functional import rotate as rotate_tensor
import cv2
import os

import numpy as np

from model import deeper_Unet_like, vanilla_Unet
from model_deconv import vanilla_Unet2

from utils.blobs_utils import get_boxes, a_link_to_the_past, get_local_maxima
from video_display_dataloader import get_video_dataloaders
from utils.grid_utils import get_landmarks_positions, get_faster_landmarks_positions, \
    get_homography_from_points, conflicts_managements, display_on_image

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


if __name__ == '__main__':
    torch.cuda.empty_cache()
    size = (256, 256)

    threshold = 0.75

    epochs = 100

    save_projection = True
    # save_projection = False

    field_length = 50
    markers_x = np.linspace(0, field_length, 11)
    field_width = 25
    lines_y = np.linspace(0, field_width, 11)

    path = 'pool model.pth'
    model = vanilla_Unet(final_depth=len(markers_x) + len(lines_y))

    batch_size = 64

    models_path = 'models'

    model_path = os.path.join(models_path, path)
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

    img_path = 'images/test_image.jpg'
    img = io.imread(img_path)
    # img = self.zoom_out(img)
    img = cv2.resize(img, size)

    tensor_img = img_transform(img)
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



        img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(img, batch_out, threshold,
                                                                          write_on_image=True,
                                                                          lines_nb=len(lines_y),
                                                                          markers_x=markers_x, lines_y=lines_y)
        cv2.imwrite('images/test_output.jpg', img)
