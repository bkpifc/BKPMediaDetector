# BKP Media Detector
This application allows you to apply various pre-trained DeepLearning Models (Tensorflow object detection / OpenVINO object detection / Face Recognition / Age & Gender estimation, Speech Recognition). It does so in a folder full of multimedia (image, video, audio) files. The input consists of a folder which contains all sorts of media files. The script then generates an output file with the hash-values (MD5) and the score of the media files where the system detected one of the "classes" in one of the applied models. It encompasses a GUI to select the input folder, the output folder, the models to apply as well as the requested output format for further use with review tools. Selectable output format for ingestion into Nuix or X-Ways available.

Currently applied models:
- Open Images (Inception-ResNet, SSD) - Detecting Objects, Persons, Animals - 300ms per image
- AVA Model (F-RCNN ResNet 101) - Detection actions and interactions - 91ms per image
- ISLogoDetector (ResNet 50) - Detecting IS Logos - 89ms per image
- FaceDetection (MTCNN) - Detecting Faces - 90ms per image
- FaceRecognition - Recognizing known Faces - <100ms per image
- OpenVINO Object Detection (VGG19) - Recognizing 1000 classes of ImageNet - 25ms per image
- OpenVINO Age & Gender estimation - Estimating age & gender of detected faces - 25ms per image
- pyAudioAnalysis Speech Recognition - Determine whether the audio file contains speech (vs. sound/noise)

## Data structure:
- Repository
  - Script (BKPMediaDetector.py)
  - Requirements.txt
- Models via Releases 


## Dependencies & Installation:
- Ubuntu 16.04 / 18.04 OR Windows 10 
- Installed Tensorflow with ObjectDetection API: [Installation Instructions](https://github.com/tensorflow/models/tree/master/research/object_detection)
- Installed OpenVINO framework: [Installation Instructions](https://github.com/opencv/dldt)
- Python 3.5 (including Numpy, magic, PySimpleGUI, PIL, OpenCV, ImageHash, pyAudioAnalysis) - 
- Windows Only: Installed VisualStudio Build Tools (Windows Support experimental)
- Windows Only: Download & compile the protocol buffers in Version 3.4 from [here](https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0) and install them by navigating to the "tensorflow/models/research" folder and executing the following command: 

`protoc.exe object_detection\protos\*.proto --python_out=.`

## Usage:
1) Download the script and the model files via [latest model release](https://github.com/bkpifc/BKPMediaDetector/releases)

2) Prepare a directory containing images and videos to be processed

3) Adjust the script and update the variable "PATH_TO_OBJECT_DETECTION_DIR" with the path where you stored the "tensorflow/models/research" folder containing the object detection API

4) Execute BKPMediaDetector.py. In order to activate the OpenVINO environment variables, you have to first execute the setupvars command in bash

`source /opt/intel/openvino/bin/setupvars.sh`
`./BKPMediaDetector.py`

## Tips:
- Make sure to use Tensorflow-GPU to leverage your GPU
- Prepare your environment to make best use of CUDA, cuDNN and your GPU. Check Tensorflows build configuration guide to validate versions.
- For help configuring your environment, just google your OS version including CUDA, cuDNN and Tensorflow as well as OpenVINO
- Getting it to work requires admittedly lots of preconfiguration and dependency management - sorry for that :-/ Docker container in progress - until then, run the script in Pycharm or similar and see what dependencies to resolve.

For guidance on re-training the models, or to create a fully new tensorflow model, please see those various excellent postings or contact us directly: 

[Step by step tensorflow object detection api tutorial](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)

[TF Object Detection Model Training](https://gist.github.com/douglasrizzo/c70e186678f126f1b9005ca83d8bd2ce)

## License:
Release under Apache 2.0 license.
Using input from Tensorflow (https://github.com/tensorflow/), the Open Images Project (https://storage.googleapis.com/openimages/web/extras.html) as well as the AVA Project (https://research.google.com/ava/download.html) all together released under Apache 2.0. Whereas no modification was done to the pre-trained model files, the actual detection script is my own work, relying though on certain python modules.
Face Detector and Face Recognition Modules are released under MIT license, based on David Sandberg's adaption of Facenet/MTCNN as well as Adam Geitgey's Face_Recognition work. ImageHash is released under BSD 2-Clause by Johannes Buchner.
OpenVINO is released under Apache 2.0 by Intel Corporation.
pyAudioAnalysis is released under Apache 2.0 by Theodoros Giannakopoulos (tyiannak)

Feel free to use and adapt - feedback is appreciated, no warranty/liability.
