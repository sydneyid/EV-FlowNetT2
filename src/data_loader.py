#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import math

_MAX_SKIP_FRAMES = 6
_TEST_SKIP_FRAMES = 4
_N_SKIP = 1

def safe_load_image(filename):
    filename = filename.numpy().decode('utf-8')
    if not os.path.exists(filename):
        # Return a dummy tensor with shape [0] to signal missing file
        return tf.constant([], dtype=tf.uint8)
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    return image

def safe_load_image_tf(filename, channels=1):
    def _safe_load(filename_np):
        # filename_np is either bytes or a numpy array
        if isinstance(filename_np, bytes):
            filename_str = filename_np.decode('utf-8')
        else:
            filename_str = filename_np.numpy().decode('utf-8')
        if not os.path.exists(filename_str):
            return np.zeros((0, 0, channels), dtype=np.uint8)
        image = tf.io.read_file(filename_str)
        image = tf.image.decode_png(image, channels=channels)
        return image.numpy()
    image = tf.py_function(_safe_load, [filename], tf.uint8)
    image.set_shape([None, None, channels])
    return image

def find_next_valid_img_path(idx, root_path, prefix, cam, channels=1, max_search=10):
    import os
    for offset in range(max_search):
        idx_str = str(idx + offset).zfill(5)
        path = os.path.join(root_path.numpy().decode('utf-8'), prefix.numpy().decode('utf-8'), f"{cam.numpy().decode('utf-8')}_image{idx_str}.png")
        if os.path.exists(path):
            return path
    # If not found, return empty string
    return ""

def build_img_path_tf(idx, root_path, prefix, cam, channels=1):
    path = tf.py_function(find_next_valid_img_path, [idx, root_path, prefix, cam, channels], tf.string)
    return path

def load_image_wrapper(filename):
    image = tf.py_function(safe_load_image, [filename], tf.uint8)
    image.set_shape([None, None, None])  # Allow dynamic shape
    return image

def filter_missing(image, *rest):
    # Filter out images with shape [0]
    return tf.size(image) > 0

def rotate_image(image, angle_degrees):
    # Convert degrees to radians
    angle_rad = angle_degrees * math.pi / 180.0

    # Get image shape
    image_shape = tf.shape(image)
    height = tf.cast(image_shape[0], tf.float32)
    width = tf.cast(image_shape[1], tf.float32)

    # Calculate center of image
    cx, cy = width / 2.0, height / 2.0

    # Build the rotation matrix
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    # Translation to rotate around center
    rotation_matrix = tf.stack([
        [cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy],
        [sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx],
        [0.0,    0.0,   1.0]
    ])

    # Create transformation for tf.raw_ops.ImageProjectiveTransformV2
    flat_matrix = tf.reshape(rotation_matrix, [-1])[:8]

    # Add batch dimension
    image = tf.expand_dims(image, 0)

    # Transform (use NEAREST for speed or BILINEAR for better quality)
    transformed = tf.raw_ops.ImageProjectiveTransformV2(images=image,
        transforms=tf.expand_dims(flat_matrix, 0),
        output_shape=tf.shape(image)[1:3], interpolation="BILINEAR")

    return tf.squeeze(transformed, 0)

def read_file_paths(data_folder_path, split, sequence=None):
    tfrecord_paths = []
    n_ima = 0
    if sequence is None:
        with open(os.path.join(data_folder_path, f"{split}_bags.txt"), 'r') as bag_list_file:
            lines = bag_list_file.read().splitlines()
    else:
      lines = sequence if isinstance(sequence, (list, )) else [sequence]
    for line in lines:
        bag_name = line
        with open(os.path.join(data_folder_path, bag_name, 'n_images.txt'), 'r') as num_ima_file:
            num_imas_split = num_ima_file.read().split(' ')
        n_left_ima = int(num_imas_split[0]) - _MAX_SKIP_FRAMES
        n_ima += n_left_ima
        tfrecord_paths.append(os.path.join(data_folder_path, bag_name, "left_event_images.tfrecord"))
        n_right_ima = int(num_imas_split[1]) - _MAX_SKIP_FRAMES
        if n_right_ima > 0 and split != 'test':
            n_ima += n_right_ima
            tfrecord_paths.append(os.path.join(data_folder_path, bag_name, "right_event_images.tfrecord"))
    return tfrecord_paths, n_ima


def _parse_function(serialized_example, image_width, image_height, skip_frames, time_only, count_only, split, root_path):
    features = {
        'image_iter': tf.io.FixedLenFeature([], tf.int64),
        'shape': tf.io.FixedLenFeature([], tf.string),
        'event_count_images': tf.io.FixedLenFeature([], tf.string),
        'event_time_images': tf.io.FixedLenFeature([], tf.string),
        'image_times': tf.io.FixedLenFeature([], tf.string),
        'prefix': tf.io.FixedLenFeature([], tf.string),
        'cam': tf.io.FixedLenFeature([], tf.string)
    }
    data = tf.io.parse_single_example(serialized_example, features)
    shape = tf.io.decode_raw(data['shape'], tf.uint16)
    shape = tf.cast(shape, tf.int32)

    event_count_images = tf.io.decode_raw(data['event_count_images'], tf.uint16)
    event_count_images = tf.reshape(event_count_images, shape)
    event_count_images = tf.cast(event_count_images, tf.float32)

    event_time_images = tf.io.decode_raw(data['event_time_images'], tf.float32)
    event_time_images = tf.reshape(event_time_images, shape)
    event_time_images = tf.cast(event_time_images, tf.float32)

    image_times = tf.io.decode_raw(data['image_times'], tf.float64)

    if split == 'test':
        n_frames = 1 #_TEST_SKIP_FRAMES if skip_frames else 1
    else:
        n_frames = tf.random.uniform([], 1, _MAX_SKIP_FRAMES, dtype=tf.int64) * _N_SKIP

    timestamps = tf.stack([image_times[0], image_times[n_frames]])

    event_count_image = tf.reduce_sum(event_count_images[:n_frames, :, :, :], axis=0)
    event_time_image = tf.reduce_max(event_time_images[:n_frames, :, :, :], axis=0)
    event_time_image = event_time_image / tf.maximum(tf.reduce_max(event_time_image), 1e-6) # time stamp is between 0 and 1

    if count_only:
        event_image = event_count_image
    elif time_only:
        event_image = event_time_image
    else:
        event_image = tf.concat([event_count_image, event_time_image], axis=2)
    event_image = tf.cast(event_image, tf.float32)

    image_iter = data['image_iter']
    prefix = data['prefix']
    cam = data['cam']

    def build_img_path(idx):
        idx_str = tf.strings.as_string(idx, width=5, fill='0') #tf.strings.format("{:05d}", [idx2])
        return tf.strings.join([root_path, "/", prefix, "/", cam, "_image", idx_str, ".png"])
    
    prev_img_path = build_img_path_tf(image_iter, root_path, prefix, cam)
    next_img_path = build_img_path_tf(image_iter + n_frames, root_path, prefix, cam)

    prev_image = safe_load_image_tf(prev_img_path, channels=1)
    prev_image = tf.cast(prev_image, tf.float32)

    next_image = safe_load_image_tf(next_img_path, channels=1)
    next_image = tf.cast(next_image, tf.float32)

    # Early exit if any image is missing
    if tf.size(prev_image) == 0 or tf.size(next_image) == 0:
        # Return dummy tensors so filter_missing can remove this sample
        event_image = tf.zeros([image_height, image_width, 4], dtype=tf.float32)
        prev_image = tf.zeros([image_height, image_width, 1], dtype=tf.float32)
        next_image = tf.zeros([image_height, image_width, 1], dtype=tf.float32)
        timestamps = tf.zeros([2], dtype=tf.float64)
        return event_image, prev_image, next_image, timestamps


    # Data augmentation
    n_split = 6
    event_size = 4
    if time_only or count_only:
        n_split = 4
        event_size = 2

    images_concat = tf.concat([event_image, prev_image, next_image], axis=2)
    if split == 'train':
        images_concat = tf.image.random_flip_left_right(images_concat)
        random_angle = tf.random.uniform([], minval=-30, maxval=30, dtype=tf.float32)
        images_rotated = rotate_image(images_concat, random_angle)
        image_cropped = tf.image.random_crop(images_rotated, [image_height, image_width, n_split])
        event_image, prev_image, next_image = tf.split(image_cropped, [event_size, 1, 1], axis=2)
    else:
        event_image = tf.image.resize_with_crop_or_pad(event_image, image_height, image_width)
        prev_image = tf.image.resize_with_crop_or_pad(prev_image, image_height, image_width)
        next_image = tf.image.resize_with_crop_or_pad(next_image, image_height, image_width)

    
    event_image.set_shape([image_height, image_width, event_size])
    prev_image.set_shape([image_height, image_width, 1])
    next_image.set_shape([image_height, image_width, 1])
    print('event image is shape '+str(event_image.shape))

    return event_image, prev_image, next_image, timestamps

def filter_missing(event_image, prev_image, next_image, timestamps):
    # Skip samples where any image is empty
    return (tf.size(prev_image) > 0) & (tf.size(next_image) > 0)


def get_loader(root, batch_size, image_width, image_height, split=None, shuffle=True, sequence=None, skip_frames=False, time_only=False, count_only=False):
    print("Loading data!")
    if split is None:
        split = 'train'
    print('split is ' + str(split) + ' sequence is ' + str(sequence))
    tfrecord_paths_np, n_ima = read_file_paths(root, split, sequence)
    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths_np)
    parse_fn = lambda x: _parse_function(x, image_width, image_height, skip_frames, time_only, count_only, split, root)
    dataset = raw_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(filter_missing)
    if shuffle and split == 'train':
        dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, n_ima
