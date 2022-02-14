import numpy as np

class FrameLabel:
    def __init__(self, label):
        self.label = label

        self.color = []
        self.depth = []
        self.accel = []
        self.gyro = []

    def append_frame(self, color, depth, accel, gyro):
        self.color.append(color)
        self.depth.append(depth)
        self.accel.append(accel)
        self.gyro.append(gyro)

    def build_arrays(self):
        arrays = []
        arrays.append(('color', np.array(self.color)))
        arrays.append(('depth', np.array(self.depth)))
        arrays.append(('accel', np.array(self.accel)))
        arrays.append(('gyro', np.array(self.gyro)))
        return arrays

    def reset(self):
        self.color = []
        self.depth = []
        self.accel = []
        self.gyro = []
