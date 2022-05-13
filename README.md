# Automated classification of sidewalk accessibility using machine learning
For wheelchair users and other individuals with mobility impairments, accessible city sidewalks are critical to safely going about their day-to-day lives. Yet, a short walk through Boston reveals seemingly endless flaws in many of its sidewalks that make them inaccessible, and there is no comprehensive data that currently exists showing which sidewalks are accessible. Without this data, wheelchair users are forced to jeopardize their safety by learning through trial-and-error which areas are accessible to them. Meanwhile, the City of Boston tracks which sidewalks need repairs via an antiquated 311 system that requires residents to report individual issues, leaving a majority of accessibility barriers unaddressed. Thus, wheelchair users would greatly benefit if there were an automated method by which the City of Boston could map sidewalk accessibility. In this project, I used a wheelchair fitted with a camera to collect around 800,000 images of sidewalks in various locations around the city and labeled them according to surface type (i.e. concrete, brick, asphalt). Using that data, I built and trained a convolutional neural network that can classify the surface type of a sidewalk from a previously unseen image with 97% accuracy. This project provides a foundation by which the City of Boston could collect geolocated images of Boston sidewalks and then automatically classify the surface types of those sidewalks. Furthermore, the classifier could be used as a starting point for future classifiers, such as a model that identities surface damage, as the appearance of surface damage varies between surface types.

## Data collection
For recording footage, I used an Intel RealSense D435i camera attached to a manual wheelchair as pictured:
![Wheelchair and camera](/images/wheelchair_and_camera.jpg)

The camera was attached via a USB to a laptop that I placed in the seat of the wheelchair. The associated Python script is located under `/recording/record.py`. The script does not display the camera feed. This is done intentionally to reduce the frequency with which frames are dropped. I have also included `/recording/preview.py` which displays the camera feed, so you can check the feed looks fine before starting a recording.

The recording comes in the form of a .bag file, which is a custom file type defined by the Intel RealSense library. Recordings include color frames, depth frames, and IMU data. After recording a .bag file, you can view it using `/recording/play.py`. The file to play should be provided as a command line argument. For instance, `python play.py myfile.bag`.

The dependenices for all scripts in the `/recording` directory are listed in `/recording/requirements.txt`.

## Labeling
To assist with labeling, I created a script in `/labeling/surfaces.py`. It takes a .bag file as a command line argument. Using OpenCV, it displays a frame of the recording and then waits for the user to press 'c', 'a', or 'b' to label the frame as "concrete", "asphalt", or "brick", respectively. Pressing the escape key ends the script. Pressing any other key discards the frame and does not put it in the labeled data. Every 500 frames, an .npy file is created for each surface type and data type (i.e. color, depth, gyroscope, accelerometer) and written to `surfaces/<surface type>/<name of bag file>_<data type>_<starting frame>_<ending frame>.npy`.

For my classifier, I did not label surfaces that are not concrete, brick, or asphalt. The vast majority of sidewalks were one of those surface types, and the surface type of the few remaining sidewalks were unclear. As such, the behavior of the classifier is undefined for surfaces that are not either concrete, brick, or asphalt.

If you're a Boston University affiliate, please note that the labeling script is best used locally because using it on the SCC would require X forwarding and would be subject to lag.

The dependenices for this script are listed in `/labeling/requirements.txt`.

## Data preparation
The .npy files created by the labeling script need some additional tweaks before they are ready for use in TensorFlow. Each file has an array with many (typically 500) frames of data. Flattening the files such that there's only 1 frame per file makes creating a data generator in TensorFlow considerably easier. The script in `/preparing/flatten.py` takes two directories, an input directory and output directory, as command line arguments and flattens all .npy files within the input directory. It outputs numbered .npy files to the output directory. For instance, you could provide an input directory with 10 files containing 10 frames each and the output directory would be populated with files named `0.npy`, `1.npy`, ..., `99.npy`.

After flattening, you can use `/preparing/split.py` to randomly split the data into training, validation, and test directories. For command line arguments, it takes an input path and then pairs of arguments representing an output directory and the fraction of the data that directory should get. For instance `python split.py input/data 0.6 output/train 0.2 output/validation 0.2 output/test` would take all of the .npy files located at a hypothetical directory `input/data` and would put 60% of them in `output/train`, 20% in `output/validation`, and 20% in `output/test`. I used a 60/20/20 split for my classifier.

It went unused in my classifier but `/preparing/unused/match_and_flatten.py` contains an alternate flattening script that combines matching color and depth frames by putting depth as a fourth value for each pixel in the color frame. That is to say, it takes the 2-dimensional array of RGB values and the 2-dimensional array of depth values and combines them into a 2-dimensional array of "RGBD" values. This is a good way to analyze the color and depth frame as a single input because it preserves the spatial relationship between the pixels of the color and depth frames. I didn't end up using this script, however, because I had better performance using transfer learning on the RGB values. The output for this script works the same as the other flattening script. The command line arguments are an input directory of color frames, an input directory of depth frames, and an output directory. The color frames and depth frames should pair correctly when sorted alphabetically.

If you're a Boston University affiliate, `/preparing/prepare_data.sh` is an example Bash script that can be used to run the data preparation procedures as a batch job on the SCC. See [BU TechWeb](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/) for more info.

The dependencies for these scripts are listed in `/preparing/requirements.txt`.

## Training
The CNN training code is fairly standard Keras usage with one notable exception -- `/training/sequence.py` defines a custom Keras Sequence class. This is a form of data generator. This was neccesary because neither Tensorflow or Keras defines a data generator that is compatible with .npy files. The custom Sequence is initialized by specifying a batch size and a list of directories containing .npy files where each directory represents a class. The Sequence builds a list of all filenames in all directories, shuffles those filenames, and then generates batches in the randomized order. The batches are ran through the preprocessor for VGG16 because I used VGG16 as a base model.

`/training/model.py`, `/training/train.py`, and `/training/evaluate.py` contain the code for initializing, training, and evaluating the model, respectively. They are in separate files because of an issue I encountered while training on the SCC. After around 30 epochs of training, the batch job would be killed with an OOM error. This doesn't make sense because memory usage shouldn't increase with the number of epochs. I suspect that the issue is either a memory leak where Kera's training API is not properly releasing the NumPy array batches passed by my custom Sequence, or a problem with TensorFlow arbitrarily trying to acquire as much memory as it can on the SCC node. I didn't have time to fully diagnose the issue, so I ran training 10 epochs at a time to circumvent the issue.

The model architecture is defined in `/training/model.py`. It is a transfer learning model with VGG16 as the base and just a dropout layer and fully-connected layer as the top. I experimented with training a model from scratch but performance with transfer learning was far better. I also tried fine-tuning the base model after training the top layers, but I couldn't improve performance.

If you are a Boston University affiliate, `/training/train.sh` is an example Batch script that can be used to run the full training pipeline as a batch job on the SCC.

If you are a Spark! affiliate and have access to my project on the SCC, you will be able to find the trained model saved in the `/training/model` directory. 

## Results
My model achieved 97.45% accuracy on 23,000 test images.

## Additional note for Spark! affiliates
All of my footage (around 8 hours) is available in my SCC project under the directory `data`. If you have picked up this project in the future and have any questions for me, please don't hesitate to email me at zachbodi@bu.edu. I would love to see this work continued and eventually put into production.
