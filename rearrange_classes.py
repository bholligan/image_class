import pandas as pd
import numpy as np
import re
import shutil
from PIL import Image
from glob import glob
import os

# Move Yukata to one folder, and the evening gown, dress, and cocktail to another.

# Add file path
df = pd.read_csv("/ebs/user05/Annotations/Annotation_PerImage_All.csv")

# Get picture file names
def pic_num(text):
    return re.search(r'.com/\w+/(\d+)_', text).group(1) + '.jpg'

df['Filename'] = df.PictureURL.apply(pic_num)
fash1 = df[(df['Majority Q1'] == 'yes') & (df['Majority Q4'] ==
                                     'onepeople')].Filename

other = df[df['Majority Q4'] != 'onepeople'].Filename

np.random.shuffle(fash1.values)
np.random.shuffle(other.values)

# Test/train split
test_size = .3
fash1_test_count = int(len(fash1)*test_size)
other_test_count = int(len(other)*test_size)

fash1_files_test = fash1[:fash1_test_count]
fash1_files_train = fash1[fash1_test_count:]
other_files_test = other[:other_test_count]
other_files_train = other[other_test_count:]

# Move files to appropriate directories
photo_path = "/ebs/user05/Fashion10000/Photos/*"
img_folders = glob(photo_path)

img_files = []
for folder in img_folders:
    img_files += glob(folder + "/*")

base_dst = "/ebs/user05/data/"
count = 0
for img_file in img_files:
    try Image.open(img_file):
        picfile = os.path.basename(img_file)
        if picfile in fash1_files_train.values:
            shutil.copy(img_file, base_dst + "train/fashion/" + picfile)
        elif picfile in fash1_files_test.values:
            shutil.copy(img_file, base_dst + "test/fashion/" + picfile)
        elif picfile in other_files_train.values:
            shutil.copy(img_file, base_dst + "train/not_fashion/" + picfile)
        elif picfile in other_files_test.values:
            shutil.copy(img_file, base_dst + "test/not_fashion/" + picfile)
        else:
            pass
    except IOError:
        count += 1
        os.remove(img_file)
print "Files Removed", count
