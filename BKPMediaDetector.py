#!/usr/bin/env python3

######
# General Detector
# 28.03.2019
# LRB
# Adapted from Tensorflow Object Detection Sample Script
######


import numpy as np
import os
import sys
import tensorflow as tf
import hashlib
import cv2
import magic
import PySimpleGUI as sg
import csv
from itertools import groupby
from distutils.version import StrictVersion
from PIL import Image
from datetime import datetime
from multiprocessing import Pool


startTime = datetime.now()

######
#
# Collecting parameters via GUI
#
######

sg.ChangeLookAndFeel('BluePurple')

layout = [[sg.Text('Please specify the folder holding the media data:')],
          [sg.Input(), sg.FolderBrowse('Browse',initial_folder='/home')],
          [sg.Text('Where shall I place the results?')],
          [sg.Input(), sg.FolderBrowse('Browse',initial_folder='/home')],
          [sg.Text('Which things do you want to detect?')],
          [sg.Checkbox('Objects/Persons', size=(15,1)),
           sg.Checkbox('Actions'),
           sg.Checkbox('IS Logos')],
          [sg.Text('Output Format:'), sg.Listbox(values=('Nuix', 'XWays', 'csv'), size=(30, 3))],
          [sg.OK(), sg.Cancel()]]

layout_progress = [[sg.Text('Detection in progress')],
                   [sg.ProgressBar(10, orientation='h', size=(20, 20), key='progressbar')],
                   [sg.Cancel()]]


gui_input = sg.Window('BKP Media Detector').Layout(layout).Read()

error = False

for element in gui_input[1]:
    if element == '':
        error = True

if error == True:
    sg.Popup('You have not populated all fields. Aborting!',title='Error',button_color=('black', 'red'),background_color=('grey'))
    exit()



######
#
# Model and Variable Preparation
#
######

if sg.OneLineProgressMeter('BKP Media Detector', 1, 8,  'key', 'Initializing variables & parameters...',orientation='h',size=(100, 10)) == False: exit()

# Variable to determine minimum GPU Processor requirement & to disable TF log output
#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Validating TF version
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Defining multiple needed variables based on config file paths & adding object_detection directory to path
PATH_TO_TEST_IMAGES_DIR = gui_input[1][0]
PATH_TO_RESULTS = gui_input[1][1]
PATH_TO_OBJECT_DETECTION_DIR = '/home/b/Programs/tensorflow/models/research'
IMAGENAMES = os.listdir(PATH_TO_TEST_IMAGES_DIR)
TEST_IMAGE_PATHS = [PATH_TO_TEST_IMAGES_DIR + '/' + i for i in IMAGENAMES]
REPORT_FORMAT = gui_input[1][5]
sys.path.append(PATH_TO_OBJECT_DETECTION_DIR)
frames_per_second = 0.25 #frames to analyze per second of video duration

# Check which models to apply and load their label maps
from object_detection.utils import label_map_util
graphlist = []
indexlist = []

MODEL1 = bool(gui_input[1][2])
if MODEL1:
    OPEN_IMAGES_GRAPH = 'Models/OpenImages/openimages.pb'
    OPEN_IMAGES_LABELS = OPEN_IMAGES_GRAPH[:-3] + '.pbtxt'
    OPEN_IMAGES_INDEX = label_map_util.create_category_index_from_labelmap(OPEN_IMAGES_LABELS)
    graphlist.append(OPEN_IMAGES_GRAPH)
    indexlist.append(OPEN_IMAGES_INDEX)

MODEL2 = bool(gui_input[1][3])
if MODEL2:
    AVA_GRAPH = 'Models/AVA/ava.pb'
    AVA_LABELS = AVA_GRAPH[:-3] + '.pbtxt'
    AVA_INDEX = label_map_util.create_category_index_from_labelmap(AVA_LABELS)
    graphlist.append(AVA_GRAPH)
    indexlist.append(AVA_INDEX)

MODEL3 = bool(gui_input[1][4])
if MODEL3:
    SPECIAL_DETECTOR_GRAPH = 'Models/ISLogos/frozen_inference_graph.pb'
    SPECIAL_DETECTOR_LABELS = SPECIAL_DETECTOR_GRAPH[:-3] + '.pbtxt'
    SPECIAL_DETECTOR_INDEX = label_map_util.create_category_index_from_labelmap(SPECIAL_DETECTOR_LABELS)
    graphlist.append(SPECIAL_DETECTOR_GRAPH)
    indexlist.append(SPECIAL_DETECTOR_INDEX)


######
#
# Worker function to prepare and reshape the input images into a Numpy array
# and to calculate the MD5 hashes of them.
#
######

def load_image_into_numpy_array(image_path):

    try:
        # Open, measure and convert image to RGB channels
        image = Image.open(image_path)
        (im_width, im_height) = image.size

        image = image.convert('RGB')
        np_array = np.array(image.getdata()).reshape(
                    (im_height, im_width, 3)).astype(np.uint8)
        image.close()

        # Hash the image in byte-chunks of 4096
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        f.close()
        hashvalue = hash_md5.hexdigest()


        return hashvalue, np_array


    # Throw errors to stdout
    except IOError:
        magictype = str(magic.from_file(image_path, mime=True))

        # If image file cannot be read, check if it is a video
        if magictype[:5] == 'video':
            # If so, return a video flag instead of numpy array
            flag = "VIDEO"
        else:
            image_path = "Could not open image: " + str(image_path) + " (" + str(magictype) + ")\n"
            flag = "ERROR"

        return image_path, flag


    except:
        logfile.write("General error with file: " + str(image_path) + " (" + str(magictype) + ")\n")


######
#
# Worker function to prepare and reshape the input videos to a Numpy array
# and to calculate the MD5 hashes of them.
# The function analyzes as much frames as indicated in the variable "frames_per_second" (Default = 0.5)
#
######

def load_video_into_numpy_array(image_path):

    videoframes = []

    # Loading the video via the OpenCV framework
    try:
        vidcap = cv2.VideoCapture(image_path)
        im_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Calculating frames per second, total frame count and analyze rate
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyze_rate = int(framecount / fps * frames_per_second)
        if 0 < analyze_rate < 100:
            int(analyze_rate)
        elif analyze_rate >= 100:
            analyze_rate = 100 #Limiting maximum frames per video
        elif analyze_rate <= 0:
            videoerror = 'Unable to extract frames from video: ' + str(image_path) + '\n'
            return videoerror



        # Hashing the image once
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
             for chunk in iter(lambda: f.read(4096), b""):
                 hash_md5.update(chunk)
        hashvalue = hash_md5.hexdigest()

        # Extracting the frames from the video
        for percentile in range(0, analyze_rate): #analyze_rate):

            vidcap.set(cv2.CAP_PROP_POS_FRAMES, (framecount / analyze_rate) * percentile)
            success, extracted_frame = vidcap.read()

            extracted_frame = cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2RGB)

            # And reshape them into a numpy array
            np_array = np.array(extracted_frame).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

            cluster = hashvalue, np_array
            videoframes.append(cluster)

        vidcap.release()

        return videoframes

    except cv2.error:
        videoerror = 'Could not process video: ' + str(image_path) + '\n'
        return videoerror

    except:
        videoerror = 'General error processing video: ' + str(image_path) + '\n'
        return videoerror



######
#
# Detection within loaded images
# Creation of output file with hashes, detection scores and class
#
######

def run_inference_for_multiple_images(images, hashvalues):

    # Initiate variables
    detectedLogos = 0
    errorcount = 0

    # Prepare results file with headers
    detectionr = open(PATH_TO_RESULTS + "/Detection_Results.csv", 'w')
    if REPORT_FORMAT[0] == 'Nuix':
        detectionr.write("tag,searchterm\n")
    else:
        detectionr.write("hash,score,category\n")
    detectionr.flush()

    for y in range(0, len(graphlist)):

        # Create TF Session with loaded graph
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            logfile.write("*" + str(datetime.now()) + ": Starting detection with model " + str(y + 1) + " of " + str(len(graphlist)) + "*\n")

            # Update progress indicator
            if sg.OneLineProgressMeter('BKP Media Detector', 5 + y, 8, 'key', 'Detecting with model {}'.format(graphlist[y]),orientation='h',size=(100, 10)) == False: exit()

            # Load the respective detetion graph from file
            with tf.gfile.GFile(graphlist[y], 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Create TF session
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_scores', 'detection_classes'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


                # Setting the detection limit of the different models.
                if "ISLogo" not in graphlist[y]:
                    detectionlimit = 0.5
                else:
                    detectionlimit = 0.90

                # Loading the label map of the correspondig graph
                category_index = indexlist[y]

                # Conduct actual detection within single image
                for index, image in enumerate(images):
                    try:
                        output_dict = sess.run(tensor_dict,
                                               feed_dict={image_tensor: np.expand_dims(image, 0)})

                        # all outputs are float32 numpy arrays, so convert types as appropriate
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]
                        detectionhit = output_dict['num_detections']
                        output_dict['detection_classes'] = output_dict['detection_classes'][0]

                        hashvalue = hashvalues[index]

                        # Validate against detection limit (default: 65%) and write hash/score if above
                        for j in range(detectionhit):
                            score = output_dict['detection_scores'][j]
                            category = category_index[output_dict['detection_classes'][j]]

                            # Validate against the preconfigured minimum detection assurance and write to result file
                            if (score >= detectionlimit):
                                scorestring = str(score)
                                if REPORT_FORMAT[0] == 'Nuix':
                                    line = ",".join([category['name'], "md5:" + hashvalue])
                                else:
                                    line = ",".join([hashvalue, scorestring, category['name']])
                                detectionr.write(line + "\n")

                    except tf.errors.InvalidArgumentError:
                        errorcount += 1
                        logfile.write("Unable to process file dimensions of file with hash: " + str(hashvalue) + "\n")

                logfile.write("*" + str(datetime.now()) + ": Finished detection with model " + str(y + 1) + "*\n")

    detectionr.flush()
    detectionr.close()

    return detectedLogos, errorcount

######
#
# Split the report file to allow seamless integration into XWays Hash Database per category
#
######

def createXWaysReport():

    for key, rows in groupby(csv.reader(open(PATH_TO_RESULTS + "/Detection_Results.csv")),
                             lambda row: row[2]):


        # Replace special characters in categories
        if str(key) != 'category':
            key = str(key).replace("/","-")
            key = str(key).replace(".", "")
            key = str(key).replace("(", "")
            key = str(key).replace(")", "")

            # Create a separate txt file for every detected category
            with open(PATH_TO_RESULTS + "/%s.txt" % key, 'a') as rf:

                for row in rows:
                    rf.write(row[0] + "\n")

                rf.flush()

    # Get a list of all files in results directory
    resultsfiles = os.listdir(PATH_TO_RESULTS)

    # Prepend them with MD5 for seamless import into XWays
    for file in resultsfiles:
        line = "md5"
        if file[-3:] == 'txt':
            with open(PATH_TO_RESULTS + '/' + file, 'r+') as ff:
                content = ff.read()
                ff.seek(0,0)
                ff.write(line.rstrip('\r\n') + '\n' + content)


######
#
# Main program function which first loads images and then starts detection
#
######

if sg.OneLineProgressMeter('BKP Media Detector', 2, 8,  'key', 'Process started. Loading images...',orientation='h',size=(100, 10)) == False: exit()


# Create logfile
logfile = open(PATH_TO_RESULTS + "/Logfile.txt", 'w')
logfile.write('***DETECTION LOG***\n')
logfile.write("*" + str(datetime.now()) + ': Process started. Loading images...*\n')

# Prevent execution when externally called
if __name__ == '__main__':

    # Initiate needed variables
    vidlist = []
    final_images = []
    errors = []

    # Multiprocess the image load function on all CPU cores available
    pool = Pool(maxtasksperchild=100)
    processed_images = pool.map(load_image_into_numpy_array, TEST_IMAGE_PATHS, chunksize=10)
    pool.close()

    # Synchronize after completion
    pool.join()
    pool.terminate()

    # Clean the result for None types (where image conversion failed)
    processed_images = [x for x in processed_images if x != None]

    # Check for the video flag
    for processed_image in processed_images:

        if str(processed_image[1]) == "VIDEO":
            # If present, populate the video list
            vidlist.append(processed_image[0])
        elif str(processed_image[1]) == "ERROR":
            errors.append(processed_image[0])
        else:
            # If not, put it to the final images list
            final_images.append(processed_image)

    for error in errors:
        logfile.write(error)
    logfile.flush()

    # Count the number of images before adding the videoframes
    number_of_images = len(final_images)

    # Update the progress indicator
    if sg.OneLineProgressMeter('BKP Media Detector', 3, 8, 'key', 'Loading Videos...',orientation='h',size=(100, 10)) == False: exit()

    # Multiprocess the video load function on all CPU cores available
    pool = Pool(maxtasksperchild=100)
    videoframes = pool.map(load_video_into_numpy_array, vidlist)
    pool.close()

    # Synchronize after completion
    pool.join()
    pool.terminate()

    number_of_videos = 0

    # Clean the result for None types (where video conversion failed)
    for video in videoframes:
        if type(video) is str:
            errors.append(video)
        if type(video) is list:
            final_images.extend(video)
            number_of_videos =+ 1

    for error in errors:
        logfile.write(error)
    logfile.flush()

    # Split the result from the loading function into hashes and image arrays
    hashvalues, image_nps = zip(*final_images)

    # Update the progress indicator & logfile
    if sg.OneLineProgressMeter('BKP Media Detector', 4, 8, 'key', 'Starting detection...',orientation='h',size=(100, 10)) == False: exit()
    logfile.write("*" + str(datetime.now()) + ": Loading completed. Detecting...*\n")

    # Execute detection
    detectedLogos, errorcount = run_inference_for_multiple_images(image_nps, hashvalues)

    # Check whether an Xways report needs to be created
    if REPORT_FORMAT[0] == 'XWays':
        createXWaysReport()

    # Write process statistics to logfile
    logfile.write("*Results: " + PATH_TO_RESULTS + "/Detection_Results.csv*\n")
    logfile.write("*Total Amount of Files: " + str(len(TEST_IMAGE_PATHS)) + " (of which " + str(number_of_images + number_of_videos) + " where processed.)*\n")
    logfile.write("*Processed Images: " + str(number_of_images) + "*\n")
    logfile.write("*Processed Videos: " + str(number_of_videos) + " (analyzed " + str(frames_per_second) + " frames per second, up to max. 100)*\n")
    logfile.write("*Applied models:\n")
    for y in range(0, len(graphlist)): logfile.write(graphlist[y] + "\n")
    logfile.write("*Processing time: " + str(datetime.now() - startTime) + "*\n")
    logfile.flush()
    logfile.close()

    # Update progress indicator
    sg.OneLineProgressMeter('BKP Media Detector', 8, 8, 'key', 'Detection finished',orientation='h',size=(100, 10))

# Deliver final success pop up to user
sg.Popup('The detection was successful',
         'The results are placed here:',
         'Path: "{}"'.format(PATH_TO_RESULTS))


