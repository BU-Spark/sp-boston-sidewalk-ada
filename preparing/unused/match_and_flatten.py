import numpy as np
import sys
import os

def match(in_path_color, in_path_depth):
    files_color = [os.path.join(in_path_color, filename) for filename in os.listdir(in_path_color) if ".npy" in filename]
    files_depth = [os.path.join(in_path_depth, filename) for filename in os.listdir(in_path_depth) if ".npy" in filename]
    files_color.sort()
    files_depth.sort()
    return zip(files_color, files_depth)

def flatten(matched_files, out_path):
    num_elems = 0
    for file_color, file_depth in matched_files:
        data_color = np.load(file_color)
        data_depth = np.load(file_depth)
        for image_color, image_depth in zip(data_color, data_depth):
            if num_elems == 50000:
                exit(0) # Cap number of output files at 50,000

            depth_shape = (image_color.shape[0], image_color.shape[1], 1)
            image_depth_reshaped = np.reshape(image_depth, depth_shape)
            output_image = np.concatenate((image_color, image_depth_reshaped), axis=2)
            np.save(os.path.join(out_path, f"{num_elems}.npy"), output_image)
            num_elems += 1

def main(args):
    in_path_color, in_path_depth, out_path = args[0], args[1], args[2]
    matched_files = match(in_path_color, in_path_depth)
    flatten(matched_files, out_path)

if __name__ == '__main__':
    main(sys.argv[1:])

