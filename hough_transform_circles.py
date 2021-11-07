import os
import time
import json
from collections import defaultdict
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_images(path_data: str, files: List[str], configs: dict) -> None:
    """Processes images based on Hough transform with the passed
    configurations.

    :param path_data: Path to the input images
    :param files: Name of the input files
    :param configs: Configurations for the Hough transforms
    :return:
    """
    assert len(files) == len(configs)
    print(f'Processing {len(files)} images...')

    for file in tqdm(files):
        conf = configs[file]
        img = cv2.imread(os.path.join(path_data, file), cv2.IMREAD_GRAYSCALE)

        thr1, thr2 = conf['edge_thr1'], conf['edge_thr2']
        img_thr = cv2.Canny(img, thr1, thr2)

        circles = detect_circles_hough(img_thr, **conf)

        file_name = f"{file.split('.png')[0]}_result.png"
        path_save = os.path.join(os.getcwd(), 'results', file_name)
        plot_result(img, img_thr, circles, path_save)


def detect_circles_hough(
    img_thr: np.ndarray,
    r_min: int,
    r_max: int,
    steps: int,
    threshold: float,
    **kwargs
) -> List[tuple]:
    """Detects circles based on the passed parameters using the Hough
    transform.
    
    :param img_thr: Edge-detection processed image
    :param r_min: Min. radius for circles to detect
    :param r_max: Max. radius for circles to detect
    :param steps: Quantization steps for the range of [0, 2*pi]
    :param threshold: Threshold for number of points that must support a circle
                      to be detected
    :param kwargs: Additional keyword arguments
    :return: Parameters of detected circles (x, y, r)
    """
    points = []
    circles = []

    for r in range(r_min, r_max + 1):
        for t in range(steps):
            points.append((r, int(r * np.cos(2 * np.pi * t / steps)),
                           int(r * np.sin(2 * np.pi * t / steps))))

    acc = defaultdict(int)
    detected_points = list(tuple(zip(*np.nonzero(img_thr))))

    for x, y in tqdm(detected_points):
        for r, dx, dy in points:
            a = x - dx
            b = y - dy
            acc[(a, b, r)] += 1

    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k

        if v / steps >= threshold and all(
                (x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in
                circles):
            circles.append((x, y, r))

    return circles


def plot_result(
    image: np.ndarray,
    img_thr: np.ndarray,
    circles: List[tuple],
    path_save: str = None
) -> None:
    """Plots the results on a grid [image, edge-detected image,
    accumulator image, image with circles]

    :param image: Original image
    :param img_thr: Edge-detected image
    :param circles: Detected circles by Hough transform
    :param path_save: Path to save the resulting image if needed
    :return:
    """
    img_res = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    acc_img = np.zeros(image.shape, dtype=np.uint8)
    acc_img = cv2.cvtColor(acc_img, cv2.COLOR_GRAY2RGB)

    cxs, cys, crs = zip(*circles)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4,
                                             figsize=(20, 10))

    ax1.imshow(image, cmap='gray')
    ax2.imshow(img_thr, cmap='gray')
    ax3.imshow(acc_img)
    ax3.scatter(cys, cxs, c='r', s=1)

    for cx, cy, cr in zip(cxs, cys, crs):
        c = plt.Circle(xy=(cy, cx), radius=cr, linewidth=2, fill=False,
                       edgecolor='red')
        ax4.add_artist(c)

    ax4.imshow(img_res)
    plt.setp((ax1, ax2, ax3, ax4), xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0.05)

    if path_save:
        plt.savefig(path_save)

    plt.show()
    time.sleep(3)


if __name__ == '__main__':

    path_cwd = os.path.abspath(os.path.dirname(__file__))
    path_configs = os.path.join(path_cwd, 'hough_configs.json')

    with open(path_configs, 'r') as file:
        configs = json.load(file)

    path_data = os.path.join(path_cwd, '3_hough')
    files = os.listdir(path_data)
    process_images(path_data, files, configs)
