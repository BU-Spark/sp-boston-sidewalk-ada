import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys

from label import FrameLabel

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_PATH, 'surfaces')

def vector_as_array(vector):
    return np.array([vector.x, vector.y, vector.z])

def write_to_disk(labels, starting_frame, current_frame, filename):
    for label in labels:
        label_output_path = os.path.join(OUTPUT_PATH, label)
        output_arrays = labels[label].build_arrays()
        for frame_type, array in output_arrays:
            output_filename = f'{filename}_{frame_type}_{starting_frame}_{current_frame}'
            np.save(os.path.join(label_output_path, output_filename), array)
        labels[label].reset()

def main(args):
    filename = args[0]

    config = rs.config()
    rs.config.enable_device_from_file(config, filename, repeat_playback=False)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align_to = rs.stream.color
    align = rs.align(align_to)

    starting_frame = 0 if len(args) < 2 else int(args[1])
    current_frame = starting_frame
    print(f'Starting at frame {starting_frame}')

    labels = {
            'concrete': FrameLabel('concrete'),
            'brick': FrameLabel('brick'),
            'asphalt': FrameLabel('asphalt'),
            'dirt': FrameLabel('dirt'),
            'tile': FrameLabel('tile'),
            'gravel': FrameLabel('gravel')
    }

    try:
        for i in range(starting_frame):
            frames = pipeline.wait_for_frames()

        frames_labeled = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_frames.keep()

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            accel_frame = aligned_frames[2].as_motion_frame()
            gyro_frame = aligned_frames[3].as_motion_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            accel_data = vector_as_array(accel_frame.get_motion_data())
            gyro_data = vector_as_array(gyro_frame.get_motion_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET)

            if depth_colormap.shape != color_image.shape:
                resized_color_image = cv2.resize(color_image,
                        dsize=(depth_colormap.shape[1], depth_colormap.shape[0]),
                        interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            pressed_key = chr(cv2.waitKey(0))

            if (current_frame - starting_frame) % 500 == 0:
                print('Writing to disk. Please wait...')
                write_to_disk(labels, starting_frame, current_frame, filename)
                starting_frame = current_frame
                print('Done. Please continue...')

            if pressed_key in 'cC':
                print(f'{current_frame}: CONCRETE')
                labels['concrete'].append_frame(color_image, depth_image, accel_data, gyro_data)
            elif pressed_key in 'aA':
                print(f'{current_frame}: ASPHALT')
                labels['asphalt'].append_frame(color_image, depth_image, accel_data, gyro_data)
            elif pressed_key in 'bB':
                print(f'{current_frame}: BRICK')
                labels['brick'].append_frame(color_image, depth_image, accel_data, gyro_data)
            elif pressed_key in 'gG':
                print(f'{current_frame}: GRAVEL')
                labels['gravel'].append_frame(color_image, depth_image, accel_data, gyro_data)
            elif pressed_key in 'tT':
                print(f'{current_frame}: TILE')
                labels['tile'].append_frame(color_image, depth_image, accel_data, gyro_data)
            elif pressed_key in 'dD':
                print(f'{current_frame}: DIRT')
                labels['dirt'].append_frame(color_image, depth_image, accel_data, gyro_data)
            elif pressed_key == chr(27): #ESC
                break
            else:
                print('SKIPPED FRAME!')

            current_frame += 1

    finally:
        print('Writing to disk. Please wait...')
        write_to_disk(labels, starting_frame, current_frame, filename)
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv[1:])
