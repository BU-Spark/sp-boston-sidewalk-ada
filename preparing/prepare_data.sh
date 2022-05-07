#!/bin/bash -l

module load python3/3.7.10
module load tensorflow/2.7.0
python3 flatten.py labels/asphalt/color batches/asphalt/color
python3 split.py batches/asphalt/color 0.6 batches/asphalt/color/train 0.2 batches/asphalt/color/validation 0.2 batches/asphalt/color/test
python3 split.py batches/brick/color 0.6 batches/brick/color/train 0.2 batches/brick/color/validation 0.2 batches/brick/color/test
python3 split.py batches/concrete/color 0.6 batches/concrete/color/train 0.2 batches/concrete/color/validation 0.2 batches/concrete/color/test

