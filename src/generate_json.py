import json
import os

import hyps

###########################################################

# calculate cropped images' top-left coordinate
coordinates = []
for i in range(0, hyps.img_height - hyps.cropped_img_size + hyps.window_size, hyps.window_size):
    for j in range(0, hyps.img_width - hyps.cropped_img_size + hyps.window_size, hyps.window_size):
        coordinates.append([j, i])

###########################################################

root_path = 'label_path/'

label_path = [os.path.join(root_path, i) for i in os.listdir(root_path)]

dict = {}
last_img_name = ""
for path in label_path:
    with open(path, 'r') as f:
        labels = f.read().split('\n')
        labels = [x.split(",") for x in labels]
        f.close()
    
    file_name = path.split('/')[-1]
    img_name = file_name.split('_')[0]
    cropped_img_id = file_name.split('_')[1].split('.')[0]

    if last_img_name != img_name:
        last_img_name = img_name
        if dict != {}:
            with open(root_path + img_name + '.json', 'w') as f:
                json.dump(dict, f)
            dict = {}
    
    if not file_name in dict:
        dict[file_name] = {}
        dict[file_name]["images"] = {}
        dict[file_name]["images"]["id"] = img_name
        dict[file_name]["images"]["img_name"] = os.path.join(root_path.replace("label", "image"), img_name)
        dict[file_name]["images"]["img_place"] = "?"
        dict[file_name]["annotations"] = []
    
    for label in labels:
        temp = {}
        cls, x, y, w, h, p = label
        temp["img_id"] = img_name
        temp["category_id"] = int(cls)
        temp["bbox"] = []
        temp["new_bbox"] = [int(x), int(y), int(w), int(h)]
        temp["new_ori"] = coordinates[int(cropped_img_id)-1]
        temp["[probability]"] = float(p)
        dict[file_name]["annotations"].append(temp)
