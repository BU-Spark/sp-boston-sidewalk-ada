import numpy as np
import sys
import os

def flatten(in_path, out_path):
    files = os.listdir(in_path)

    num_elems = 0
    for filename in files:
        data = np.load(os.path.join(in_path, filename))
        for elem in data:
            if num_elems == 50000:
                exit(0) # Cap number of files at 50,000
            np.save(os.path.join(out_path, f"{num_elems}.npy"), elem)
            num_elems += 1

def main(args):
    in_path, out_path = args[0], args[1]
    flatten(in_path, out_path)

if __name__ == '__main__':
    main(sys.argv[1:])

