import math
import os
import random
import sys

def main(args):
    in_path = args[0]
    files = [filename for filename in os.listdir(in_path) if ".npy" in filename]
    total_files = len(files)
    random.shuffle(files)

    splits = []
    for i in range(1, len(args), 2):
        splits.append((float(args[i]), args[i+1]))

    for ratio, out_path in splits:
        for i in range(math.floor(ratio * total_files)):
            filename = files.pop()
            os.rename(os.path.join(in_path, filename), os.path.join(out_path, filename))

if __name__ == '__main__':
    main(sys.argv[1:])
