import pyrealsense2 as rs
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCESSIBLE_DIR = os.path.join(BASE_DIR, 'data/accessible')
INACCESSIBLE_DIR = os.path.join(BASE_DIR, 'data/inaccessible')

config = rs.config()
rs.config.enable_device_from_file(config, 'test.bag')
config.enable_stream(rs.stream.color, rs.format.rgb8, 15)
config.enable_stream(rs.stream.depth, rs.format.z16, 15)

pipeline = rs.pipeline()
pipeline.start(config)

try:
    num_accessible = 0 
    num_inaccessible = 0
    while True:
        # Get the next available color and depth frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert these frames to numpy arrays (representing images)
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create a colormapped version of the depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET)

        # If depth and color are different shapes, resize color. Then, horizontally stack them.
        if depth_colormap.shape != color_image.shape:
            resized_color_image = cv2.resize(color_image,
                    dsize=(depth_colormap.shape[1], depth_colormap.shape[0]),
                    interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Display the frame and wait for a key press
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        pressed_key = chr(cv2.waitKey(0))

        # 1 = label as accessible, 2 = label as inaccessible, ESC = quit
        if pressed_key == '1':
            cv2.imwrite(os.path.join(ACCESSIBLE_DIR, f'color_{num_accessible}.jpeg'), color_image)
            cv2.imwrite(os.path.join(ACCESSIBLE_DIR, f'depth_{num_accessible}.jpeg'), depth_image)
            num_accessible += 1
        elif pressed_key == '2':
            cv2.imwrite(os.path.join(INACCESSIBLE_DIR, f'color_{num_inaccessible}.jpeg'), color_image)
            cv2.imwrite(os.path.join(INACCESSIBLE_DIR, f'depth_{num_inaccessible}.jpeg'), depth_image)
            num_inaccessible += 1
        elif pressed_key == chr(27): #ESC
            break

finally:
    pipeline.stop()
