from glob import glob
import os
import pickle
import shutil
from tqdm import tqdm
import bson

import random
import numpy as np

all_val_pics = glob("/workspace6/miha_misc2/val/**/*") # Folder with
for img in tqdm(all_val_pics):
    destination = img.replace("val", "train")
    shutil.move(img, destination)

d_info = []

with open('/workspace6/miha_misc2/train.bson', 'rb') as fbson:
    data = bson.decode_file_iter(fbson)
    for c, d in tqdm(enumerate(data)):
        category = d['category_id']
        _id = d['_id']
        d['imgs']
        d_info.append([category, _id, len(d['imgs'])])


random.seed(0)
random.shuffle(d_info)

val1 = d_info[:50000]
val2 = d_info[50000:100000]
d_info = d_info[100000:]

s_order = pickle.load(open("./data/class_order.pkl", "rb"))
s_order = {int(x): i for i, x in enumerate(s_order)}


def generate_info_dict(in_list, savename):
    info = {}
    info["num_class"] = 5270
    classes = []
    filenames = []

    for category, xid, n_img in in_list:
        for idx in range(n_img):
            classes.append(s_order[category])
            sname = os.path.join(str(category), str(xid) + "-{}.jpg".format(idx))
            filenames.append(sname)

    info["filenames"] = filenames
    info["classes"] = np.array(classes).astype(np.int32)
    info["samples"] = len(filenames)
    info["num_class"] = 5270
    pickle.dump(info, open(savename, "wb"))

generate_info_dict(val1, "generator_val1_v1.pkl")
generate_info_dict(val2, "generator_val2_v1.pkl")
generate_info_dict(d_info, "generator_train_v1.pkl")

