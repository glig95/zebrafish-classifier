import numpy as np 
from pathlib import Path
import imageio 
from tqdm.auto import tqdm
import tensorflow as tf
import albumentations
from types import SimpleNamespace


def create_dataset(folder, target_size, batchsize=16, augment=False, shuffle=True, n_tiles=1):
    transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=.5),
        albumentations.VerticalFlip(p=.5),          
        albumentations.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=.1, val_shift_limit=0, p=.5),
        albumentations.RandomBrightnessContrast(brightness_limit = (-0.3,0.3), p=.5),
#         albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=15, border_mode=0, value=(1,1,1), p=.5),        
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=20, p=.5),                
        albumentations.GaussianBlur(p=.3),        
#         albumentations.Affine(scale=(0.8,1.2), shear=0, rotate=(-10,10), cval=(1,1,1), p=.5),
#         albumentations.RandomBrightnessContrast(brightness_limit=.1, contrast_limit=.1, p=.5),
#         albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=.1, val_shift_limit=0, p=.5)
#         albumentations.GaussianBlur(p=.3),
    ])
    
    if not all(s%n_tiles==0 for s in target_size):
        raise ValueError(f'Unsupported value {n_tiles=} - all dimensions of target_size {target_size} must be divisble by n_tiles!') 
        
    def aug_fn(x):
        x = np.stack([transform(image=_x)['image'] for _x in x])
        return x

    def normalize(x,y):
        x = x/255.
        return x,y
        
    def process_aug(x,y):
        x = tf.numpy_function(func=aug_fn, inp=[x], Tout=tf.float32)
        return x, y
    
    def tile(x,y):
        x = tf.concat(tf.split(tf.stack(tf.split(x, n_tiles, axis=0), 0), n_tiles, axis=2),0)
        y = tf.stack([y]*n_tiles**2)
        return x,y

    data = tf.keras.utils.image_dataset_from_directory(folder, batch_size=batchsize, shuffle=shuffle, image_size=target_size)
    
    filenames = data.file_paths
    
    class_name_to_id = dict((v,k) for k,v in enumerate(data.class_names))
    class_id_to_name = dict((k, v) for k,v in enumerate(data.class_names))

    # normalize 
    data = data.map(normalize)

    data = data.unbatch().map(tile).unbatch()
    data = data.apply(tf.data.experimental.assert_cardinality(n_tiles**2*len(filenames)))
    data = data.batch(batchsize)

    labels = np.concatenate(tuple(y.numpy() for x, y in data), axis=0)
    images = np.concatenate(tuple(x.numpy() for x, y in data), axis=0)

    if augment:
        data = data.map(process_aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return SimpleNamespace(data=data, images=images, labels=labels, filenames=filenames, class_name_to_id=class_name_to_id, class_id_to_name=class_id_to_name)


# +
import matplotlib.pyplot as plt 
data = create_dataset('data/validation/', augment=True, batchsize=16, target_size=(450,900), n_tiles=1)

n=4
x,y = next(iter(data.data))
plt.figure(figsize=(20,6))
for i, (_x, _y) in enumerate(zip(x[:2*n],y[:2*n])):
    plt.subplot(2,n, i+1)
    plt.imshow(np.clip(_x, 0,1))
    plt.title(f'New image - class = {data.class_id_to_name[int(_y)]} ({int(_y)})')

