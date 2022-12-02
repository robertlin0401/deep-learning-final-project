import os
import cv2

from label_processor import read_label_file
from label_processor import node_in_bbox, get_cropped_bbox

import hyps

###########################################################

# constants
label_train = [os.path.join('data/labels/train', i) for i in os.listdir('data/labels/train')]
label_val = [os.path.join('data/labels/val', i) for i in os.listdir('data/labels/val')]
label_test = [os.path.join('data/labels/test', i) for i in os.listdir('data/labels/test')]

###########################################################

# calculate cropped images' top-left coordinate
coordinates = []
for i in range(0, hyps.img_height - hyps.cropped_img_size + hyps.window_size, hyps.window_size):
    for j in range(0, hyps.img_width - hyps.cropped_img_size + hyps.window_size, hyps.window_size):
        coordinates.append([j, i])

###########################################################

# crop images
for label_dir in [label_train, label_val, label_test]:
    for path in label_dir:
        annotations = []
        for i in range(len(coordinates)):
            annotations.append({
                "cls": [],
                "bbox":[]
            })

        f = open(path, 'r')
        labels = read_label_file(f)
        f.close()

        for label in labels:
            if not int(label[3]) > 0 or not int(label[4]) > 0:
                continue
            if not hyps.use_cls0_flag and int(label[0]) == 0:
                continue
            if not hyps.use_cls1_flag and int(label[0]) == 1:
                continue
            if not hyps.use_cls2_flag and int(label[0]) == 2:
                continue
            if not hyps.use_cls3_flag and int(label[0]) == 3:
                continue
            for i, coordinate in enumerate(coordinates):
                if node_in_bbox(label[1:], coordinate):
                    annotations[i]["cls"].append(label[0])
                    annotations[i]["bbox"].append(get_cropped_bbox(label[1:], coordinate))
        
        valid_crops = []
        for i, annotation in enumerate(annotations):
            if annotation["cls"] != []:
                valid_crops.append(i)
                with open(path.split('.')[0] + '_' + str(i+1) + '.txt', 'w') as f:
                    for j in range(len(annotation["cls"])):
                        cls = str(annotation["cls"][j])
                        bbox = [str(i) for i in annotation["bbox"][j]]
                        f.write(cls + ',' + ','.join(bbox) + '\n')
                    f.close()
            elif hyps.use_unlabeled_flag:
                valid_crops.append(i)
                with open(path.split('.')[0] + '_' + str(i+1) + '.txt', 'w') as f:
                    f.close()
        os.remove(path)

        img_path = path.replace('labels', 'images').replace('txt', 'png')
        img = cv2.imread(img_path)
        for i in valid_crops:
            crop_img = img[
                coordinates[i][1]: (coordinates[i][1] + hyps.cropped_img_size),
                coordinates[i][0]: (coordinates[i][0] + hyps.cropped_img_size)
            ]
            if len(crop_img) < hyps.cropped_img_size or len(crop_img[0]) < hyps.cropped_img_size:
                os.remove(path.split('.')[0] + '_' + str(i+1) + '.txt')
                continue
            cv2.imwrite(img_path.split('.')[0] + '_' + str(i+1) + '.png', crop_img)
        os.remove(img_path)
        
        print("Crop image " + img_path.split('/')[-1])        
