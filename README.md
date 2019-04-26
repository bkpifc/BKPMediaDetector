# BKP Media Detector
This python script allows you to apply up to 4 different object detection models (in Frozen Graph Form - .pb). It does so in a folder full of images and / or video files. The input consists of a folder which contains all sorts of media files. The script then generates an output file with the hash-values (MD5) and the score of the media files where the system detected one of the "classes" in one of the applied models. It encompasses a GUI to select the input folder, the output folder, the models to apply (4 default ones) as well as the requested output format for further use with review tools. Selectable output format for ingestion into Nuix or X-Ways available.

Currently applied models:
- Open Images (Inception-ResNet, SSD) - Detecting Objects, Persons, Animals - 300ms per image
- AVA Model (F-RCNN ResNet 101) - Detection actions and interactions - 91ms per image
- ISLogoDetector (ResNet 50) - Detecting IS Logos - 89ms per image
- FaceDetection (MTCNN) - Detecting Faces - 90ms per image

## Data structure:
- Repository
  - Script (BKPMediaDetector.py)
- Models via Releases (frozen TF graphs, pbtxt files, current version of script)


## Dependencies:
- Ubuntu 16.04 / 18.04
- Installed Tensorflow with ObjectDetection API: [Installation Instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
- Python 3.5 (including Numpy, magic, PySimpleGUI, PIL, OpenCV)

## Usage:
1) Download the script and the models files via [latest model release](https://github.com/bkpifc/BKPMediaDetector/releases)

2) Prepare a directory containing images and videos to be processed

3) Adjust the script and update the variable "PATH_TO_OBJECT_DETECTION_DIR" with the path where you stored the "tensorflow/models/research" folder containing the object detection API

4) Execute BKPMediaDetector.py in the same folder as the model folder:

`./BKPMediaDetector.py`

## Tips:
- Make sure to use Tensorflow-GPU to leverage your GPU
- Prepare your environment to make best use of CUDA, cuDNN and your GPU. Check Tensorflows build configuration guide to validate versions.
- For help configuring your environment, just google your OS version including CUDA, cuDNN and Tensorflow

For guidance on re-training the model, or to create a fully new tensorflow model, please see those various excellent postings or contact us directly: 

[Step by step tensorflow object detection api tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

[TF Object Detection Model Training](https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce)

## License:
Release under Apache 2.0 license.
Using input from Tensorflow (https://github.com/tensorflow/), the Open Images Project (https://storage.googleapis.com/openimages/web/extras.html) as well as the AVA Project (https://research.google.com/ava/download.html) all together released under Apache 2.0. Whereas no modification was done to the pre-trained model files, the actual detection script is my own work, relying though on certain python modules.
Face Detector Module is released under MIT license, based on David Sandberg's adaption of Facenet/MTCNN.
Feel free to use and adapt - feedback is appreciated.
