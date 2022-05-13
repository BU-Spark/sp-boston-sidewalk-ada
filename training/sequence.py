import random
import os
import numpy as np
import tensorflow as tf

class SurfaceDataSequence(tf.keras.utils.Sequence):
    def __init__(self, class_paths, batch_size):
        self.batch_size = batch_size
        self.num_classes = len(class_paths)
        self.files = []
        for i, path in enumerate(class_paths):
            self.files += [(os.path.join(path, filename), i) for filename in os.listdir(path) if ".npy" in filename]
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files) // self.batch_size

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            batch_x.append(np.load(self.files[idx * self.batch_size + i][0]))
            batch_y.append(self.files[idx * self.batch_size + i][1])
        return (tf.keras.applications.vgg16.preprocess_input(np.array(batch_x)),
                np.array(batch_y))
