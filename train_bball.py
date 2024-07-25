'''
modeled after trainer_XCORRELATION.py

'''
import torch
from torch import save, load
from torch.nn import MSELoss, L1Loss, BCELoss, CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

import os
from dataloader import get_train_test_dataloaders_bball
from torch.utils.tensorboard import SummaryWriter
from model_deconv import vanilla_Unet2
from tqdm.auto import tqdm

# TODO: add cmd line parser so you can specify things like epochs, etc
if __name__ == '__main__':

    torch.cuda.empty_cache()
    # device-agnostic, in practice probably wanna train on gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()  # default location is ./runs

    train_img_path = 'dataset/ncaa_bball/images'
    train_grid_path = 'dataset/ncaa_bball/grids'
    # lines_nb = 11
    lines_nb = 7  # number of vertical markers in template grid (right now we have 15x7 grid)

    # final depth needs to be 22 because we're training using 15x7 uniform grid representation of court (15+7=22)
    model = vanilla_Unet2(final_depth=22).to(device)  # initialize model

    model_prefix = ''
    best_model_path = 'best_bball.pth'
    final_model_path = 'final_bball.pth'
    batch_size = 8
    models_path = './models/'
    epochs_already_trained = 0

    size = (256, 256)
    lr = 1e-3
    epochs_nb = 5  # model will train for (epochs_nb - epochs_already_trained) epochs

    optimizer_function = Adam
    save_after_N_epochs = 5
    display_frequency = 3

    initial_temperature = 1
    stagnation = 0.95

    train_file = 'dataset/ncaa_bball/train.txt'
    train_dataloader, test_dataloader = get_train_test_dataloaders_bball(train_img_path, train_grid_path, size, train_file,
                                                                         batch_size=batch_size, train_test_ratio=0.8,
                                                                         lines_nb=lines_nb)
    train_dataloader.temperature = initial_temperature
    test_dataloader.augment_data = False
    print('dataloader and model loaded')

    optimizer = optimizer_function(model.parameters(),
                                   lr=lr,
                                   weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-5, verbose=True)

    lines_criterion = CrossEntropyLoss(ignore_index=100)
    markers_criterion = CrossEntropyLoss(ignore_index=100)
    mask_criterion = BCELoss()
    mask_coef = 2
    markers_coef = 5

    if not os.path.isdir(models_path): os.mkdir(models_path)

    # if epochs_already_trained != 0:
    #     model.load_state_dict(load(models_path + model_prefix + 'best_model.pth'))

    # load pre-trained model
    # if you've already trained a basketball model, then you can do additional training on that model
    # if you haven't done any training yet, you can use the pre-trained soccer model that's in the repo
    soccer_model_path = 'soccer model.pth'
    model.load_state_dict(load(os.path.join(models_path, soccer_model_path)))

    display_counter = 0
    prev_best_loss = 1000

    for epoch in tqdm(range(epochs_already_trained, epochs_already_trained + epochs_nb)):

        train_dataloader.temperature *= stagnation

        ### TRAIN PART ###
        total_epoch_loss = 0
        model.train()
        for batch in train_dataloader:
            img = batch['img'].to(device)
            truth = batch['out'].to(device)
            truth_mask = batch['mask'].to(device)

            out = model.forward(img)

            out_lines = out[:, :lines_nb]
            out_markers = out[:, lines_nb:]
            truth_lines = truth[:, 0]
            truth_markers = truth[:, 1]

            lines_loss = lines_criterion(out_lines, truth_lines)
            markers_loss = markers_criterion(out_markers, truth_markers)
            mask_loss = mask_criterion(torch.max(out, dim=1)[0], truth_mask)

            loss = lines_loss + \
                   markers_loss * markers_coef + \
                   mask_loss * mask_coef

            total_epoch_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            display_counter += 1
            if display_counter == display_frequency:
                display_counter = 0
                print(float(loss))
        # print(f'debug: len train_dataloader: {len(train_dataloader)}')
        total_epoch_loss /= len(train_dataloader)
        print('train :', total_epoch_loss, epoch)
        writer.add_scalar("Loss/train", total_epoch_loss, epoch)  # record loss

        ### TEST PART ###
        model.eval()
        total_epoch_loss = 0
        with torch.inference_mode():
            for batch in test_dataloader:
                img = batch['img'].to(device)
                truth = batch['out'].to(device)
                truth_mask = batch['mask'].to(device)

                out = model.forward(img)

                out_lines = out[:, :lines_nb]
                out_markers = out[:, lines_nb:]
                truth_lines = truth[:, 0]
                truth_markers = truth[:, 1]

                lines_loss = lines_criterion(out_lines, truth_lines)
                markers_loss = markers_criterion(out_markers, truth_markers)
                mask_loss = mask_criterion(torch.max(out, dim=1)[0], truth_mask)

                loss = lines_loss + \
                       markers_loss * markers_coef + \
                       mask_loss * mask_coef

                total_epoch_loss += float(loss)
            # print(f'debug: len test_dataloader: {len(test_dataloader)}')
            total_epoch_loss /= len(test_dataloader)
            writer.add_scalar("Loss/test", total_epoch_loss, epoch)  # record test loss

            print('test :', total_epoch_loss, epoch)
            if (total_epoch_loss < prev_best_loss):  # only want to save the later models to save space
                prev_best_loss = total_epoch_loss
                # save(model.state_dict(), models_path + model_prefix + f'bball_unetv2_epoch{epoch}.pth')
                save(model.state_dict(), os.path.join(models_path, best_model_path))
                print('\t\tSaved best model at epoch ' + str(epoch))
            # print()
        torch.cuda.empty_cache()
        scheduler.step(total_epoch_loss)

    # saving final model
    save(model.state_dict(), os.path.join(models_path, final_model_path))
    print(f'Saved final model')
    writer.flush()
    writer.close()

    # calculating eval metrics for best and final model
