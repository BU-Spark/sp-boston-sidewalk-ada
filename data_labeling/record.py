from datetime import datetime
import pyrealsense2 as rs

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    filename = datetime.now().strftime('%m-%d-%Y_%H:%M:%S') + '.bag'
    config.enable_record_to_file(filename)

    pipeline.start(config)

    try:
        pipeline.wait_for_frames()
        print('Recording has begun...')
        while True:
            pipeline.wait_for_frames()

    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()

