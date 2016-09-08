from keras.preprocessing import image
from glob import glob
import os

base = '/ebs/user05/data/*'

subfolders = []
files = []
for folder in glob(base):
    subfolders += glob(folder + '/*')

for folders in subfolders:
    files += glob(folders + '/*')

count = 0
for file in files:
    try:
        image.load_img(file)
    except OSError:
        count += 1
        os.remove(file)
print "Files Removed", count
