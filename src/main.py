""" Merve Gul Kantarci Vision Lab Assignment 3"""

import dataloader
import mervenet
import helper
from torch import nn
import imageio
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on:", device)
custom_model_path = "best_net.pth"
anat_path = "dataset/annotations/"

dataset_loaders = {}
dataset_sizes = {}


def get_trained_network(network, epoch_size=30):

    loss_criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    model_optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 10 epochs
    opt_lr_scheduler = lr_scheduler.StepLR(model_optimizer, step_size=10, gamma=0.1)

    model_trained = train_network(network, loss_criterion, model_optimizer, opt_lr_scheduler, epoch_size)

    return model_trained.cpu()


def train_network(model, criterion, optimizer, scheduler, num_epochs):

    print("Training starts...")

    begin = time.time()
    model.train(True)
    best_model = copy.deepcopy(model.state_dict())
    best_loss = float("Inf")
    loss_dict = {"train": [], "val": []}

    for epoch in range(num_epochs * 2):

        part = "val"
        if epoch % 2 == 0:
            part = "train"
            print('Epoch {}/{}'.format((epoch // 2) + 1, num_epochs))
            print('-' * 10)
        part_loss = 0.0

        if part == 'train':
            # # decreases learning rate
            scheduler.step()
            # Set model to training mode
            model.train(True)
            for inputs, labels in dataset_loaders[part]:
                # shuffle the data
                shuffle_inds = torch.randperm(inputs.size()[0])
                inputs_shuffled = inputs[shuffle_inds]
                labels_shuffled = labels[shuffle_inds]
                inputs = inputs_shuffled.to(device)
                labels = labels_shuffled.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # back propogation
                    loss.backward()
                    optimizer.step()
                # add loss
                part_loss += loss.item() * inputs.size(0)
        else:
            # Set model to evaluate mode
            model.eval()
            for inputs, labels in dataset_loaders[part]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                # loss
                part_loss += loss.item() * inputs.size(0)

        epoch_loss = part_loss / dataset_sizes[part]
        loss_dict[part].append(epoch_loss)

        print('{} Loss: {:.4f}'.format(part, epoch_loss))

        # deep copy the best model
        if part == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model)
    # save model
    torch.save(model, custom_model_path)
    # visualize learning
    helper.plot_graphs(loss_dict, num_epochs)

    return model


def run_tests(model, all_test_pairs, coords_dict):
    start = time.time()
    print("Testing starts...")
    model.train(False)
    model.eval()
    model.to(device)
    # this will be base directory for GIFs
    if not os.path.exists("results/"):
        os.makedirs("results/")

    for test_pairs in all_test_pairs:
        imgs = []
        loss = 0
        with torch.no_grad():
            # first pair is a bit of exception
            upscaled_coords = dataloader.get_coords_from_path(test_pairs[0][0], coords_dict)
            for pair in test_pairs:
                # coords from anat file
                true_coords1 = dataloader.get_coords_from_path(pair[0], coords_dict)
                # add loss
                loss += (np.square(np.array(upscaled_coords) - np.array(true_coords1))).mean()
                s1 = helper.draw_boxes(pair[0], upscaled_coords, true_coords1)
                features, rels, ratios = dataloader.extract_feature_single_pair(pair, upscaled_coords)
                # get estimated coordinates relative to small image
                rel_coords = model(features.to(device))
                # get coordinates relative to big image
                upscaled_coords = dataloader.upscale_coords(rel_coords.tolist(), rels, ratios)
                imgs.append(s1)
        # Block below is for the last image (since there is no next)
        true_coords1 = dataloader.get_coords_from_path(pair[1], coords_dict)
        loss += (np.square(np.array(upscaled_coords) - np.array(true_coords1))).mean()
        s1 = helper.draw_boxes(pair[1], upscaled_coords, true_coords1)
        imgs.append(s1)
        gif_path = "results/" + dataloader.get_video_no_from_path(pair[1]) + '.gif'
        imageio.mimsave(gif_path, imgs)  # save as GIF
        print("Loss for the test video is " + str(loss / len(imgs)))
        print("Results extracted to " + gif_path)
    end = time.time()
    print("Testing completed in " + str(round((end - start) / 60)) + " minutes")


if __name__ == "__main__":
    if_extract = input("Do you want to extract features? y/n \nIf this is the first run please type 'y'")
    # modify vgg16 to extract features
    dataloader.modify_model()
    # model for estimating bounding box
    init_network = mervenet.Net().to(device)
    if if_extract == "y":
        print("Features will be extracted from scratch.")
        """ for the first time, features should be extracted, below block used for this.
         It can be commented after the first extraction """
        print("Extraction starts...")
        start = time.time()
        train_img_pairs, train_anat_dict = dataloader.find_path_pairs("dataset/videos/train", anat_path, True)
        val_img_pairs, val_anat_dict = dataloader.find_path_pairs("dataset/videos/val", anat_path, True)
        dataloader.extract_features("train", train_img_pairs, train_anat_dict)
        dataloader.extract_features("val", val_img_pairs, val_anat_dict)
        end = time.time()
        print("Train and Val extraction completed in " + str(round((end - start) / 60)) + " minutes")
    else:
        print("Features will be taken from the directory.")
    # below line is enough for features after first extraction
    dataset_loaders, dataset_sizes = dataloader.load_features(["train", "val"], batch_size=256)
    # train the network
    trained = get_trained_network(init_network.to(device), epoch_size=30)
    # below line uses the network from previous run (optional)
    # trained = torch.load(custom_model_path)
    # test phase
    test_img_pairs, test_anat_dict = dataloader.find_test_path_pairs("dataset/videos/test", anat_path)
    # extract features and estimate
    run_tests(trained, test_img_pairs, test_anat_dict)
    print("Task completed.")
