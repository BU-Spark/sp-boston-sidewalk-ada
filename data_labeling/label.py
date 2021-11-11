import cv2
import numpy as np
import os
import pyrealsense2 as rs
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCESSIBLE_DIR = os.path.join(BASE_DIR, 'data/accessible')
INACCESSIBLE_DIR = os.path.join(BASE_DIR, 'data/inaccessible')

def main(argv):
    config = rs.config()
    rs.config.enable_device_from_file(config, argv[0])
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline = rs.pipeline()
    pipeline.start(config)

    try:
        frame_num = 0 
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
                cv2.imwrite(os.path.join(ACCESSIBLE_DIR, f'{argv[0]}_color_{frame_num}.jpeg'), color_image)
                cv2.imwrite(os.path.join(ACCESSIBLE_DIR, f'{argv[0]}_depth_{frame_num}.jpeg'), depth_image)
            elif pressed_key == '2':
                cv2.imwrite(os.path.join(INACCESSIBLE_DIR, f'{argv[0]}_color_{frame_num}.jpeg'), color_image)
                cv2.imwrite(os.path.join(INACCESSIBLE_DIR, f'{argv[0]}_depth_{frame_num}.jpeg'), depth_image)
            elif pressed_key == chr(27): #ESC
                break
            
            frame_num += 1

    finally:
        pipeline.stop()

if __name__ == '__main__':
    main(sys.argv[1:])
