from datetime import datetime
import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    filename = datetime.now().strftime('%m-%d-%Y_%H:%M:%S') + '.bag'
    config.enable_record_to_file(filename)

    pipeline.start(config)

    try:
        frame_num = 0
        while True:
            # Get the next available color and depth frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            print(f'received frame {frame_num}')
            frame_num += 1

    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()

