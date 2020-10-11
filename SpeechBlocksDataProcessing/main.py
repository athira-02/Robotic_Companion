import scipy.io as sio
import numpy as np
import os


data_path = '//SketchySceneData'

train_dir = '//SketchySceneData/train/CLASS_GT/'
test_dir = '//SketchySceneData/test/CLASS_GT/'
val_dir = '//SketchySceneData/val/CLASS_GT/'


test_file = open('test_data.csv', 'w')

onlyfiles = next(os.walk(test_dir))[2]
length = len(onlyfiles)
for i in range(1, length + 1):
    filename = 'sample_' + str(i) + '_class.mat'

    print('test: ' + filename)

    mat_contents = sio.loadmat(test_dir + filename)
    segmented_matrix = mat_contents['CLASS_GT']

    sprite_list = []
    spriteString = 'sample_' + str(i) + ","

    for x in np.nditer(segmented_matrix):
        if sprite_list.count(x) == 0 and x != 0:
            sprite_list.append(x)
            spriteString = spriteString + str(x) + ","

    length = len(spriteString)
    spriteString = spriteString[0:(length - 1)]  # remove the last comma
    test_file.write(spriteString + '\n')

test_file.close()

validation_file = open('validation_data.csv', 'w')

onlyfiles = next(os.walk(val_dir))[2]
length = len(onlyfiles)
for i in range(1, length + 1):
    filename = 'sample_' + str(i) + '_class.mat'

    print('val: ' + filename)

    mat_contents = sio.loadmat(val_dir + filename)
    segmented_matrix = mat_contents['CLASS_GT']

    sprite_list = []
    spriteString = 'sample_' + str(i) + ","

    for x in np.nditer(segmented_matrix):
        if sprite_list.count(x) == 0 and x != 0:
            sprite_list.append(x)
            spriteString = spriteString + str(x) + ","

    length = len(spriteString)
    spriteString = spriteString[0:(length - 1)]  # remove the last comma
    validation_file.write(spriteString + '\n')

validation_file.close()
