The sports_field_registration folder contains all the
code that is used to estimation the court homography.

A short description of how the model works can be
seen in README.md. This repo was forked from
https://github.com/njacquelin/sports_field_registration
and modified for basketball.

Below is a description of how to use the code to
create a custom training dataset, how to actually
train the model, and how to perform inference on unseen
court images.

make_dataset.py:
running make_dataset.py creates (or edits if already existing)
a court homography dataset of court images and 
corresponding homography images. It allows the user
to go through a game frame by frame and annotate
each court image.
The file structure is dataset/ncaa_bball/[annotations or images]/game name/[.npy (annotations) or .png (images)].
Additionally, a train.txt file is created that contains
the names of the games that were processed. Will have to change
input_video_path to your own path

check_bball_dataset.py:
running check_bball_dataset.py iterates through
the specified dataset and displays the transformed
version of each image using the given homography. 
This can be used as a sanity check to make sure the
dataset was created correctly.

data_management/generate_grid_template.py:
run this to create the grid.npy file that is
the 2d grid template of uniform points that represent
the court.

data_management/generate_img_grid.py:
run this to create the grid versions of the original
court images that will be used in training. The grids
will be saved in dataset/ncaa_bball/grids.

train_bball.py:
run this to train a model on the custom dataset
that can predict the homography of an unseen
court image

dataloader.py:
contains BballDataset class that loads and augments
the dataset

homography_utils.py:
contains functions that can perform inference on given
court image. These functions are used in the main
tracking data loop.


homography_demo.py:
run this to get draw predicted court lines on an input
video.
TODO: refactor code

