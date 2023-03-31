""" Merve Gul Kantarci Vision Lab Assignment 3"""

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(loss, num_epochs):
    # plot loss graph
    plt.plot(loss["train"], color='green')
    plt.plot(loss["val"], color='red')
    plt.xticks(np.arange(0, num_epochs, step=5, dtype=np.int))
    plt.legend(['train loss', 'validation loss'], loc='upper left')
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.title("Tain & Validation Loss Change in Each Epoch")
    plt.show()


def draw_boxes(path, est_coords, true_coords1):
    source_img1 = Image.open(path)
    draw = ImageDraw.Draw(source_img1)
    # draw estimated box
    draw.rectangle(est_coords)
    # draw ground truth box
    draw.rectangle(true_coords1, outline=128)  # red

    return source_img1