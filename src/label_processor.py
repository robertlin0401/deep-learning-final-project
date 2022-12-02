import hyps

###########################################################

def read_label_file(f):
    labels = f.read().split('\n')[:-1]
    labels = [x.split(",") for x in labels]
    labels = [[int(i) for i in label] for label in labels]
    return labels

###########################################################

def node_in_bbox(bbox, coordinate):
    ratio = 0.8

    x1 = coordinate[0]
    x2 = coordinate[0] + hyps.cropped_img_size
    y1 = coordinate[1]
    y2 = coordinate[1] + hyps.cropped_img_size

    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])
    
    if (x2 <= x or x + w <= x1) and (y2 <= y or y + h <= y1):
        return False
    else:
        lens = min(x2, x + w) - max(x1, x)
        wide = min(y2, y + h) - max(y1, y)
        if (lens * wide) / (w * h) > ratio:
            return True
        else:
            return False

def get_cropped_bbox(bbox, coordinate):
    x1 = coordinate[0]
    x2 = coordinate[0] + hyps.cropped_img_size
    y1 = coordinate[1]
    y2 = coordinate[1] + hyps.cropped_img_size

    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    if x < x1:
        new_x = x1
        if x + w < x2:
            new_w = x + w - x1
        else:
            new_w = hyps.cropped_img_size
    elif x > x1:
        new_x = x 
        if x + w < x2:
            new_w = w
        else:
            new_w = x2 - x
    else:
        new_x = x
        if w < hyps.cropped_img_size:
            new_w = w
        else:
            new_w = hyps.cropped_img_size
    
    if y < y1:
        new_y = y1
        if y + h < y2:
            new_h = y + h - y1
        else:
            new_h = hyps.cropped_img_size
    elif y > y1:
        new_y = y 
        if y + h < y2:
            new_h = h
        else:
            new_h = y2 - y
    else:
        new_y = y
        if h < hyps.cropped_img_size:
            new_h = h
        else:
            new_h = hyps.cropped_img_size
    
    new_x = new_x - x1
    new_y = new_y - y1
    
    return [new_x, new_y, new_w, new_h]

###########################################################

import numpy as np

def convert_label_to_yolo(labels):
    labels = [[float(i) for i in label] for label in labels]

    labels = np.copy(labels)

    labels[:, 1] = labels[:, 1] + (labels[:, 3] / 2)
    labels[:, 2] = labels[:, 2] + (labels[:, 4] / 2)

    labels[:, [1, 3]] = labels[:, [1, 3]] / hyps.cropped_img_size
    labels[:, [2, 4]] = labels[:, [2, 4]] / hyps.cropped_img_size

    return labels

def convert_label_from_yolo(labels):
    labels = np.copy(labels)

    labels[:, [1, 3]] = labels[:, [1, 3]] * hyps.cropped_img_size
    labels[:, [2, 4]] = labels[:, [2, 4]] * hyps.cropped_img_size

    labels[:, 1] = labels[:, 1] - (labels[:, 3] / 2)
    labels[:, 2] = labels[:, 2] - (labels[:, 4] / 2)

    for j in range(5):
        labels[:, j] = [int(i) for i in labels[:, j]]

    return labels

###########################################################

import matplotlib.pyplot as plt
import cv2

def plot_bounding_box(img_path, label_path):
    img = cv2.imread(img_path)
    with open(label_path, "r") as f:
        labels = read_label_file(f)

    for label in labels:
        cls, x, y, w, h = label
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    dpi = 120
    fig = plt.figure(figsize = (hyps.img_width / dpi, hyps.img_height / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img)
    plt.show()
