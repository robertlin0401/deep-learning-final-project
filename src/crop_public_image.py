import os
import cv2

import hyps

###########################################################

# constants
images = [os.path.join('public_data', i) for i in os.listdir('public_data')]

###########################################################

# calculate cropped images' top-left coordinate
coordinates = []
for i in range(0, hyps.img_height - hyps.cropped_img_size + hyps.window_size, hyps.window_size):
    for j in range(0, hyps.img_width - hyps.cropped_img_size + hyps.window_size, hyps.window_size):
        coordinates.append([j, i])

###########################################################

# crop images
for img_path in images:
    img = cv2.imread(img_path)
    for i in range(len(coordinates)):
        crop_img = img[
            coordinates[i][1]: (coordinates[i][1] + hyps.cropped_img_size),
            coordinates[i][0]: (coordinates[i][0] + hyps.cropped_img_size)
        ]
        if len(crop_img) < hyps.cropped_img_size or len(crop_img[0]) < hyps.cropped_img_size:
            continue
        cv2.imwrite(img_path.split('.')[0] + '_' + str(i+1) + '.png', crop_img)
    os.remove(img_path)
    
    print("Crop image " + img_path.split('/')[-1])        
