import numpy as np
import tensorflow as tf
from sequence import SurfaceDataSequence

TRAIN_CLASS_PATHS = ('batches/concrete/color/train',
        'batches/asphalt/color/train',
        'batches/brick/color/train')
VALIDATION_CLASS_PATHS = ('batches/concrete/color/validation',
        'batches/asphalt/color/validation',
        'batches/brick/color/validation')

train_data = SurfaceDataSequence(TRAIN_CLASS_PATHS, 32)
validation_data = SurfaceDataSequence(VALIDATION_CLASS_PATHS, 32)

model = tf.keras.models.load_model('model')
model.layers[0].trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
model.fit(train_data,
        epochs=10,
        validation_data=validation_data,
        verbose=2,
        class_weight={0: 1.00, 1: 1.78, 2: 1.36})

model.save('model')
