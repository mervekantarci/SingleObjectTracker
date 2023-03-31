""" Merve Gul Kantarci Vision Lab Assignment 3"""

import os
from glob import glob
import random
import torch
from torchvision import models, transforms
from torch import nn
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get pretrained Model for extracting features
extractor = (models.vgg16(pretrained=True)).to(device)

scale_amount = 20
transform_comp = transforms.Compose([
    transforms.Resize(scale_amount),
    transforms.ToTensor(),
    # to [-1, +1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def find_path_pairs(video_path, anat_path, shuffle):
    """" Due to memory issues images will be opened in feature extraction part """

    coordinates = {}
    img_paths = []
    for root in glob(video_path + "/*/"):
        video_no = os.path.basename(os.path.normpath(root))
        prev_path = ""
        coordinates[video_no] = dict()
        with open(anat_path + video_no + ".ann") as anat:
            for line in anat:
                splitted_anat = line.strip().split(" ")
                coordinates[video_no][int(splitted_anat[0])] = list(splitted_anat[1:])
        for i, file in enumerate(glob(root + "*.jpg")):
            img_paths.append((prev_path, file))
            prev_path = file
        del img_paths[len(img_paths) - i - 1]  # remove the empty string concatenated pair

    if shuffle:
        # data will be shuffled in training anyway
        random.shuffle(img_paths)
    return img_paths, coordinates


def find_test_path_pairs(video_path, anat_path):
    """" Due to memory issues images will be opened in feature extraction part.
    Very similar to train and validation reading but list structure is different """

    coordinates = {}
    img_paths = []
    for root in glob(video_path + "/*/"):
        video_no = os.path.basename(os.path.normpath(root))
        prev_path = ""
        coordinates[video_no] = dict()
        video_img_paths = []
        with open(anat_path + video_no + ".ann") as anat:
            for line in anat:
                splitted_anat = line.strip().split(" ")
                coordinates[video_no][int(splitted_anat[0])] = list(splitted_anat[1:])
        for i, file in enumerate(glob(root + "*.jpg")):
            video_img_paths.append((prev_path, file))
            prev_path = file
        del video_img_paths[len(video_img_paths) - i - 1]  # remove the empty string concatenated pair
        img_paths.append(video_img_paths)

    return img_paths, coordinates


def get_2x_bounding_box(img, coordinate_list):
    """" Crops the image in 2x enlarged b. box
    and returns the new top left corner to calculate relative coordinates"""

    x1 = float(coordinate_list[0])
    y1 = float(coordinate_list[1])
    x2 = float(coordinate_list[2])
    y2 = float(coordinate_list[3])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    nx1 = center_x - ((center_x - x1) * 2)
    nx2 = center_x - ((center_x - x2) * 2)
    ny1 = center_y - ((center_y - y1) * 2)
    ny2 = center_y - ((center_y - y2) * 2)

    # incase it exceeds the image boundaries
    if nx1 < 0:
        nx1 = 0
    if nx2 > img.size[0]:
        nx2 = img.size[0]
    if ny1 < 0:
        ny1 = 0
    if ny2 > img.size[1]:
        ny2 = img.size[1]

    # crop image
    imgn = img.crop((nx1, ny1, nx2, ny2))

    return imgn, nx1, ny1


def get_4x_bounding_box(img, coordinate_list):
    """" Crops the image in 4x enlarged b. box
    and returns the new top left corner to calculate relative coordinates
    This is needed because bounding box gets too small rarely. Deatils in report.
    """

    x1 = float(coordinate_list[0])
    y1 = float(coordinate_list[1])
    x2 = float(coordinate_list[2])
    y2 = float(coordinate_list[3])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    nx1 = center_x - ((center_x - x1) * 4)
    nx2 = center_x - ((center_x - x2) * 4)
    ny1 = center_y - ((center_y - y1) * 4)
    ny2 = center_y - ((center_y - y2) * 4)

    # in cse it exceeds the image boundaries
    if nx1 < 0:
        nx1 = 0
    if nx2 > img.size[0]:
        nx2 = img.size[0]
    if ny1 < 0:
        ny1 = 0
    if ny2 > img.size[1]:
        ny2 = img.size[1]

    imgn = img.crop((nx1, ny1, nx2, ny2))

    return imgn, nx1, ny1


def get_transformed_image_pair(pair, anats):

    video_no = os.path.basename(os.path.normpath(os.path.dirname(pair[0])))

    img1 = Image.open(pair[0])
    img2 = Image.open(pair[1])
    img1_no = int(os.path.basename(pair[0])[:-4])
    img2_no = int(os.path.basename(pair[1])[:-4])
    img1_cropped, corner_x1, corner_y1 = get_2x_bounding_box(img1, anats[video_no][img1_no])
    img2_cropped, corner_x2, corner_y2 = get_2x_bounding_box(img2, anats[video_no][img1_no])
    # resize and normalize
    trnsf1 = transform_comp(img1_cropped)
    trnsf2 = transform_comp(img2_cropped)

    rel_coords = [corner_x2, corner_y2] * 2
    # the ratio of resizing , differs in each dimension x and y
    # dimension order is opposite compared to PIL in tensor
    ratios = [trnsf2.size()[2] / img2_cropped.size[0], trnsf2.size()[1] / img2_cropped.size[1]] * 2
    true_coords = [float(x) for x in anats[video_no][img2_no]]
    # obtain the coordinates acc to processed image
    rel_scaled_true_coords = [(x - y) * z for x, y, z in zip(true_coords, rel_coords, ratios)]

    return trnsf1, trnsf2, rel_scaled_true_coords


def get_transformed_test_image_pair(pair, coord_list):

    img1 = Image.open(pair[0])
    img2 = Image.open(pair[1])
    img1_cropped, corner_x1, corner_y1 = get_2x_bounding_box(img1, coord_list)
    img2_cropped, corner_x2, corner_y2 = get_2x_bounding_box(img2, coord_list)

    # when where to look frame is really small, avoid from vanishing
    w, h = img2_cropped.size
    if w < 10 or h < 10:
        img2_cropped, corner_x2, corner_y2 = get_4x_bounding_box(img2, coord_list)
    w, h = img1_cropped.size
    if w < 10 or h < 10:
        img1_cropped, corner_x1, corner_y1 = get_4x_bounding_box(img1, coord_list)

    rel_coords = [corner_x2, corner_y2] * 2
    # resize and normalize
    trnsf1 = transform_comp(img1_cropped)
    trnsf2 = transform_comp(img2_cropped)
    # the ratio of resizing , differs in each dimension x and y
    ratios = [trnsf2.size()[2] / img2_cropped.size[0], trnsf2.size()[1] / img2_cropped.size[1]] * 2

    return trnsf1, trnsf2, rel_coords, ratios


def modify_model():
    """" Cast the extarctor """
    features_modified = nn.Sequential(*list(extractor.features.children()))
    features_modified[-1] = torch.nn.modules.pooling.AvgPool2d(512)  # 512 to lower the dimensions to 1
    extractor.features = features_modified


def upscale_coords(coords, rels, ratios):
    """" Get coordinates for original image size """
    upscaled_coords = [(x / z) + y for x, y, z in zip(coords, rels, ratios)]
    return upscaled_coords


def extract_feature_single_pair(pair, coord_list):
    """" This is used for testing while each frame pair feature is extracted individually"""

    # set model to evaluation mode
    extractor.train(False)
    extractor.eval()

    with torch.no_grad():
        data1, data2, rels, ratios = get_transformed_test_image_pair(pair, coord_list)
        output1 = extractor.features(data1.unsqueeze(0).to(device)).reshape(-1)
        output2 = extractor.features(data2.unsqueeze(0).to(device)).reshape(-1)
        features = torch.cat([output1, output2])

    return features, rels, ratios


def extract_features(dataset_type, pair_list, anats_dict):
    """" Extract features for each pair """
    features = []
    label_list = []  # true (processed) labels

    # set model to evaluation mode
    extractor.train(False)
    extractor.eval()

    with torch.no_grad():
        for pair in tqdm(pair_list):
            data1, data2, true_coordinates = get_transformed_image_pair(pair, anats_dict)
            label_list.append(true_coordinates)
            output1 = extractor.features(data1.unsqueeze(0).to(device)).reshape(-1)
            output2 = extractor.features(data2.unsqueeze(0).to(device)).reshape(-1)
            combined = torch.cat([output1, output2])
            features.append(combined)

    print("Extraction completed for " + dataset_type + " features.")
    features_combined = torch.stack(features)
    labels_combined = torch.tensor(label_list)
    # save for future use
    torch.save(features_combined, dataset_type + "_featurestensor.pt")
    torch.save(labels_combined, dataset_type + "_labelstensor.pt")
    print("Fature tensors are saved. Feature size is ", features_combined.size())
    # features and relative coordinates for processed image
    return features, label_list


def load_features(dataset_types, batch_size=256):
    """" Loads already extracted features """
    dataset_loaders = {}
    dataset_sizes = {}

    for type in dataset_types:
        fs = torch.load(type + "_featurestensor.pt")
        ls = torch.load(type + "_labelstensor.pt")
        # make batches
        fs_batched = torch.split(fs, batch_size)
        ls_batched = torch.split(ls, batch_size)
        # make batches
        dataset_loaders[type] = [(fs_chunk, ls_chunk) for fs_chunk, ls_chunk in zip(fs_batched, ls_batched)]
        dataset_sizes[type] = ls.size()[0]

    return dataset_loaders, dataset_sizes


def get_coords_from_path(path, anat_dict):
    """" Returns coords from anat file """
    video_no = get_video_no_from_path(path)
    img_no = int(os.path.basename(path)[:-4])

    return [float(x) for x in anat_dict[video_no][img_no]]


def get_video_no_from_path(path):
    """" Returns video no """
    video_no = os.path.basename(os.path.normpath(os.path.dirname(path)))
    return video_no


