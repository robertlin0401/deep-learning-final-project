import matplotlib.pyplot as plt
import os

from label_processor import read_label_file
from label_processor import convert_label_from_yolo, convert_label_to_yolo

###########################################################

train = [os.path.join('data/labels/train', i) for i in os.listdir('data/labels/train')]
val = [os.path.join('data/labels/val', i) for i in os.listdir('data/labels/val')]
test = [os.path.join('data/labels/test', i) for i in os.listdir('data/labels/test')]

for j in [train, val, test]:
    for i in j:
        with open(i, 'r+') as f:
            labels = read_label_file(f)
            labels = convert_label_to_yolo(labels)
            labels = [[str(i) for i in label] for label in labels]
            f.seek(0)
            for label in labels:
                f.write(' '.join(label) + '\n')
            f.close()
        print("convert label " + i.split('/')[-1])
