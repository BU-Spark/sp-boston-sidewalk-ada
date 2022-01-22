import cv2
import numpy as np
import pyrealsense2 as rs
import sys

def main(argv):
    config = rs.config()
    rs.config.enable_device_from_file(config, argv[0])
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    pipeline = rs.pipeline()
    pipeline.start(config)

    try:
        while True:
            # Get the next available color and depth frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            accel_frame = frames[2].as_motion_frame()
            gyro_frame = frames[3].as_motion_frame()

            # Convert these frames to numpy arrays (representing images)
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            accel_data = np.asanyarray(accel_frame.get_motion_data())
            gyro_data = np.asanyarray(gyro_frame.get_motion_data())

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
            cv2.waitKey(1)
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main(sys.argv[1:])
