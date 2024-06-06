# Sports Field Registration

The `sports_field_registration` folder contains all the code used to estimate the court homography.

## Description

A short description of how the model works can be seen in `README.md`. This repo was forked from [njacquelin/sports_field_registration](https://github.com/njacquelin/sports_field_registration) and modified for basketball.

Below is a description of how to use the code to create a custom training dataset, train the model, and perform inference on unseen court images.

## Scripts and Usage

### `make_dataset.py`

Running `make_dataset.py` creates (or edits if already existing) a court homography dataset of court images and corresponding homography images. It allows the user to go through a game frame by frame and annotate each court image.

- **File Structure:** 
  - `dataset/ncaa_bball/[annotations or images]/game name/[.npy (annotations) or .png (images)]`
- **Output:** 
  - A `train.txt` file containing the names of the games processed.
  - Court images and corresponding homographies in new dataset in `dataset/ncaa_bball`
- **Note:** 
  - Change `input_video_path` to your own path.

### `check_bball_dataset.py`

Running `check_bball_dataset.py` iterates through the specified dataset and displays the transformed version of each image using the given homography. This can be used as a sanity check to ensure the dataset was created correctly.

### `data_management/generate_grid_template.py`

Run this to create the `grid.npy` file, which is the 2D grid template of uniform points representing the court.

### `data_management/generate_img_grid.py`

Run this to create the grid versions of the original court images that will be used in training. The grids will be saved in `dataset/ncaa_bball/grids`.

### `train_bball.py`

Run this to train a model on the custom dataset that can predict the homography of an unseen court image.

### `dataloader.py`

Contains the `BballDataset` class that loads and augments the dataset.

### `homography_utils.py`

Contains functions that can perform inference on a given court image. These functions are used in the main tracking data loop.

### `homography_demo.py`

Run this to draw predicted court lines on an input video.

