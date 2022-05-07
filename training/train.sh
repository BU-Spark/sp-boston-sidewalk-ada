#!/bin/bash -l

module load python3/3.7.10
module load tensorflow/2.8.0

echo "Initializing model..."
python model.py

for n in {1..5};
do
    echo "Starting training session ${n}..."
    python train.py
done

echo "Starting tuning..."
python tune.py

echo "Starting evaluation..."
python evaluate.py

