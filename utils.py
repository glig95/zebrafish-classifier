import numpy as np 
from pathlib import Path
import imageio 
from tqdm.auto import tqdm
import tensorflow as tf
import albumentations
from collections import Counter
from types import SimpleNamespace

# +
from keras.utils.image_dataset import paths_and_labels_to_dataset, ALLOWLIST_FORMATS
from keras.utils import dataset_utils
from keras.utils import image_utils

def my_image_dataset_from_directory(directory,
                                 labels='inferred',
                                 label_mode='int',
                                 class_names=None,
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear',
                                 follow_links=False,
                                 crop_to_aspect_ratio=False,
                                 **kwargs):
    

    if 'smart_resize' in kwargs:
        crop_to_aspect_ratio = kwargs.pop('smart_resize')
    if kwargs:
        raise TypeError(f'Unknown keywords argument(s): {tuple(kwargs.keys())}')
    if labels not in ('inferred', None):
        if not isinstance(labels, (list, tuple)):
            raise ValueError(
              '`labels` argument should be a list/tuple of integer labels, of '
              'the same size as the number of image files in the target '
              'directory. If you wish to infer the labels from the subdirectory '
              'names in the target directory, pass `labels="inferred"`. '
              'If you wish to get a dataset that only contains images '
              f'(no labels), pass `labels=None`. Received: labels={labels}')
    if class_names:
          raise ValueError('You can only pass `class_names` if '
                       f'`labels="inferred"`. Received: labels={labels}, and '
                       f'class_names={class_names}')
    if label_mode not in {'int', 'categorical', 'binary', None}:
        raise ValueError(
            '`label_mode` argument must be one of "int", "categorical", "binary", '
            f'or None. Received: label_mode={label_mode}')
    if labels is None or label_mode is None:
        labels = None
        label_mode = None
    if color_mode == 'rgb':
        num_channels = 3
    elif color_mode == 'rgba':
        num_channels = 4
    elif color_mode == 'grayscale':
        num_channels = 1
    else:
        raise ValueError(
            '`color_mode` must be one of {"rgb", "rgba", "grayscale"}. '
            f'Received: color_mode={color_mode}')
    interpolation = image_utils.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(validation_split, subset, True, seed)

    if seed is None:
        seed = np.random.randint(1e6)
        
    image_paths, labels, class_names = dataset_utils.index_directory(
          directory,
          labels,
          formats=ALLOWLIST_FORMATS,
          class_names=class_names,
          shuffle=True,
          seed=seed,
          follow_links=follow_links)

    if label_mode == 'binary' and len(class_names) != 2:
        raise ValueError(
            f'When passing `label_mode="binary"`, there must be exactly 2 '
            f'class_names. Received: class_names={class_names}')

    image_paths, labels = dataset_utils.get_training_or_validation_split(
      image_paths, labels, validation_split, subset)
    if not image_paths:
        raise ValueError(f'No images found in directory {directory}. '
                     f'Allowed formats: {ALLOWLIST_FORMATS}')

    dataset = paths_and_labels_to_dataset(
      image_paths=image_paths,
      image_size=image_size,
      num_channels=num_channels,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      interpolation=interpolation,
      crop_to_aspect_ratio=crop_to_aspect_ratio)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    # Include file paths for images as attribute.
    dataset.file_paths = image_paths
    return dataset


# +
def create_dataset(folder, target_size, batchsize=16, augment=False, shuffle=True, seed=42, validation_split=None, subset=None, n_tiles=1, ensure_n_classes=None, n_workers=32):
    if augment:
        p=.5
        transform = albumentations.Compose([
            albumentations.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=.05, val_shift_limit=0, p=p),
            albumentations.RandomBrightnessContrast(brightness_limit = (-0.3,0.3), p=p),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, border_mode=4, p=p),
            albumentations.GaussianBlur((3,27), p=p),
            albumentations.ElasticTransform(p=p),
            albumentations.ToGray(p=p)
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

    data = my_image_dataset_from_directory(folder, batch_size=None, 
                                        validation_split=validation_split, subset=subset, seed=seed,
                                        image_size=target_size)
    
    
    class_name_to_id = dict((v,k) for k,v in enumerate(data.class_names))
    class_id_to_name = dict((k, v) for k,v in enumerate(data.class_names))

    filenames = data.file_paths
    
    images, labels = tuple(zip(*data.as_numpy_iterator()))
    
    data = tf.data.Dataset.from_tensor_slices((np.stack(images), np.stack(labels)))
    # normalize 
    data = data.map(normalize)
    
    counts = Counter(labels)
    
    if ensure_n_classes is not None and not len(counts)==ensure_n_classes:
            raise ValueError(f'Less than {ensure_n_classes} classes sampled for data, try a different seed!')


    data = data.map(tile).unbatch()
    data = data.apply(tf.data.experimental.assert_cardinality(n_tiles**2*len(filenames)))
    
    if shuffle:
        data = data.shuffle(len(data))
    
    data = data.batch(batchsize)

    data = data.map(process_aug, num_parallel_calls=n_workers).prefetch(n_workers)

    return SimpleNamespace(data=data, transform=transform, counts=counts, images=images, labels=labels, filenames=filenames, class_name_to_id=class_name_to_id, class_id_to_name=class_id_to_name)

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
