import numpy as np 
from pathlib import Path
import imageio 
from tqdm.auto import tqdm
import tensorflow as tf
import albumentations
from collections import Counter
from types import SimpleNamespace


# +
def create_dataset(folder, target_size, batchsize=16, augment=False, shuffle=True, n_tiles=1):
    if augment:
        transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=.5),
        albumentations.VerticalFlip(p=.5),          
        albumentations.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=.05, val_shift_limit=0, p=.5),
        albumentations.RandomBrightnessContrast(brightness_limit = (-0.3,0.3), p=.5),
        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, border_mode=0, value=(1,1,1), p=.5),                
        albumentations.GaussianBlur((3,17), p=.3),        
    ])
    else:
        transform = albumentations.Compose([])
    
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

    data = tf.keras.utils.image_dataset_from_directory(folder, batch_size=batchsize, shuffle=False, image_size=target_size)
    
    filenames = data.file_paths
    
    class_name_to_id = dict((v,k) for k,v in enumerate(data.class_names))
    class_id_to_name = dict((k, v) for k,v in enumerate(data.class_names))

    # normalize 
    data = data.map(normalize)

    labels = np.concatenate(tuple(y.numpy() for x, y in data), axis=0)
    images = np.concatenate(tuple(x.numpy() for x, y in data), axis=0)
    
    counts = Counter(labels)

    data = data.unbatch().map(tile).unbatch()
    data = data.apply(tf.data.experimental.assert_cardinality(n_tiles**2*len(filenames)))
    
    if shuffle:
        data = data.shuffle(len(data))
    
    data = data.batch(batchsize)

    data = data.map(process_aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return SimpleNamespace(data=data, transform=transform, images=images, labels=labels, counts=counts, filenames=filenames, class_name_to_id=class_name_to_id, class_id_to_name=class_id_to_name)

def show_augmented(folder='data/validation/', w=4, h=3):
    data = create_dataset(folder, augment=True, batchsize=1, target_size=(450,900), n_tiles=1)    

    import matplotlib.pyplot as plt 
    plt.figure(figsize=(20,8))
    for j in range(h):
        x, y = data.images[j], data.labels[j]
        for i in range(w):
            _x = data.transform(image=x)['image'] if i>0 else x
            plt.subplot(h,w, w*j+i+1)
            plt.imshow(np.clip(_x, 0,1))
            title = f'Augmented {i}/{w-1} {data.class_id_to_name[y]}' if i>0 else f'Original {data.class_id_to_name[y]}'
            plt.title(title)
     
    
# show_augmented()
