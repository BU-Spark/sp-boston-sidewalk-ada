import numpy as np
import tensorflow as tf
from sequence import SurfaceDataSequence

TEST_CLASS_PATHS = ('batches/concrete/color/test',
        'batches/asphalt/color/test',
        'batches/brick/color/test')

test_data = SurfaceDataSequence(TEST_CLASS_PATHS, 32)

model = tf.keras.models.load_model('model')
model.evaluate(test_data, verbose=2)


