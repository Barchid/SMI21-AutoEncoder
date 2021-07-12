import os
import shutil
import random


if __name__ == '__main__':
    random.seed(9)
    source_path = os.path.join('datasets', 'training_data', 'rbg_imgs')
    images = os.listdir(source_path)

    random.shuffle(images)

    train_path = os.path.join('datasets', 'training_data', 'train')
    val_path = os.path.join('datasets', 'training_data', 'val')

    os.mkdir(train_path)
    os.mkdir(val_path)

    os.mkdir(os.path.join(train_path, 'rgb'))
    os.mkdir(os.path.join(val_path, 'rgb'))

    for i in range(len(images)):
        path = val_path if i%3==0 else train_path # 1/3 of the images in the val_path
        shutil.move(os.path.join(source_path, images[i]), os.path.join(path, 'rgb', images[i]))

    os.rmdir(source_path)