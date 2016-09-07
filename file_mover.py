import pandas as pd
import numpy as np
import re
import shutil
from glob import glob
import os

# Add file path
df = pd.read_csv("/Users/bholligan/git/image_class/user05/Annotations/Annotation_PerImage_All.csv")

# Get picture file names
def pic_num(text):
    return re.search(r'.com/\w+/(\d+)_', text).group(1) + '.jpg'

df['Filename'] = df.PictureURL.apply(pic_num)

# Filter by whether the file is fashion or not fashion
fash_files = df[df['Majority Q1'] == 'yes'].Filename
not_fash_files = df[df['Majority Q1'] == 'no'].Filename
np.random.shuffle(fash_files.values)
np.random.shuffle(not_fash_files.values)

# Test/train split
test_size = .3
fash_test_count = int(len(fash_files)*test_size)
nofash_test_count = int(len(not_fash_files)*test_size)

fash_files_test = fash_files[:fash_test_count]
fash_files_train = fash_files[fash_test_count:]
nofash_files_test = not_fash_files[:nofash_test_count]
nofash_files_train = not_fash_files[nofash_test_count:]

# Move files to appropriate directories
photo_path = "/Users/bholligan/git/image_class/user05/Fashion10000/Photos/*"
img_folders = glob(photo_path)

img_files = []
for folder in img_folders:
    img_files += glob(folder + "/*")

base_dst = "/Users/bholligan/git/image_class/data/"
for img_file in img_files:
    picfile = os.path.basename(img_file)
    if picfile in fash_files_train.values:
        shutil.copy(img_file, base_dst + "train/fashion/" + picfile)
    elif picfile in fash_files_test.values:
        shutil.copy(img_file, base_dst + "test/fashion/" + picfile)
    elif picfile in nofash_files_train.values:
        shutil.copy(img_file, base_dst + "train/not_fashion/" + picfile)
    else:
        shutil.copy(img_file, base_dst + "test/not_fashion/" + picfile)
