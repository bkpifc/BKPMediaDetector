#!/usr/bin/env python3

######
# General Detector
# 06.12.2018 / Last Update: 07.06.2019
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
import imagehash
from itertools import groupby
from distutils.version import StrictVersion
from PIL import Image
from datetime import datetime
from multiprocessing import Pool
from Models.Face import detect_face
from pathlib import Path


######
#
# Worker function to prepare and reshape the input images into a Numpy array
# and to calculate the MD5 hashes of them.
#
######

def load_image_into_numpy_array(image_path):

    try:
        image_path = str(image_path)
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
    except IOError or OSError:
        magictype = str(magic.from_file((image_path), mime=True))
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
    old_hash = None

    # Loading the video via the OpenCV framework
    try:
        vidcap = cv2.VideoCapture(image_path)
        im_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Calculating frames per second, total frame count and analyze rate
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyze_rate = int(framecount / fps * frames_per_second)
        if 0 < analyze_rate < max_frames_per_video:
            int(analyze_rate)
        elif analyze_rate >= int(max_frames_per_video):
            analyze_rate = max_frames_per_video #Limiting maximum frames per video
        elif analyze_rate <= 0:
            videoerror = 'Unable to extract frames from video: ' + str(image_path) + '\n'
            return videoerror

        # Hashing the video once
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

            if video_sensitivity > 0:
                # Compare the frame with the previous one for similarity, and drop if similar
                frame_to_check = Image.fromarray(np_array)
                new_hash = imagehash.phash(frame_to_check)
                if old_hash is None or (new_hash - old_hash > video_sensitivity):
                    cluster = hashvalue, np_array
                    videoframes.append(cluster)
                    old_hash = new_hash
            else:
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
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'w')
    if REPORT_FORMAT[0] == 'Nuix':
        detectionresults.write("tag,searchterm\n")
    else:
        detectionresults.write("hash,score,category\n")
    detectionresults.flush()

    for y in range(0, len(graphlist)):
        # Create TF Session with loaded graph
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            logfile.write("*" + str(datetime.now()) + ": Starting detection with model " + str(y + 1) + " of " + str(len(graphlist)) + "*\n")

            # Update progress indicator
            if sg.OneLineProgressMeter('BKP Media Detector', 5 + y, 9, 'key', 'Detecting with model {}'.format(graphlist[y]),orientation='h',size=(100, 10)) == False: exit()

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

                # Loading the label map of the corresponding graph
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
                                detectionresults.write(line + "\n")

                    except tf.errors.InvalidArgumentError:
                        errorcount += 1
                        logfile.write("Unable to process file dimensions of file with hash: " + str(hashvalue) + "\n")

                logfile.write("*" + str(datetime.now()) + ": Finished detection with model " + str(y + 1) + "*\n")

    # Executing Face Detector in a slightly different process if selected
    if FACE_MODEL:

        # Updating progress bar and logfile
        if sg.OneLineProgressMeter('BKP Media Detector', 8, 9, 'key', 'Detecting with Face Detector',orientation='h',size=(100, 10)) == False: exit()
        logfile.write("*" + str(datetime.now()) + ": Starting detection with face detection model*\n")

        # Applying constants as defined in Facenet
        minsize = 20
        threshold = [0.6, 0.7, 0.7]
        factor = 0.709

        # Creating different TF Session
        with tf.Session() as sess:

            # read pnet, rnet, onet models from Models/Face directory
            facemodel_path = Path('Models/Face')
            pnet, rnet, onet = detect_face.create_mtcnn(sess, str(facemodel_path))

            # Inference for all images
            for index, image in enumerate(images):
                try:
                    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
                    # If face detected, writing output to results file
                    if not len(bounding_boxes) == 0:
                        hashvalue = hashvalues[index]
                        number_of_faces = len(bounding_boxes)
                        if REPORT_FORMAT[0] == 'Nuix':
                            line = "Face,md5:" + hashvalue
                        else:
                            line = str(hashvalue) + "," + str(number_of_faces) + ",Faces"
                        detectionresults.write(line + "\n")

                except tf.errors.InvalidArgumentError:
                    errorcount += 1
                    logfile.write("Unable to detect faces in file with hash: " + str(hashvalue) + "\n")

        logfile.write("*" + str(datetime.now()) + ": Finished detection with face detection model*\n")

    detectionresults.flush()
    detectionresults.close()

    return detectedLogos, errorcount


######
#
# Split the report file to allow seamless integration into XWays Hash Database per category
#
######

def createXWaysReport():

    detectionresults_path = str(PATH_TO_RESULTS / 'Detection_Results.csv')
    xways_folder = PATH_TO_RESULTS / 'XWaysOutput'
    if not xways_folder.exists(): os.mkdir(str(xways_folder))

    for key, rows in groupby(csv.reader(open(detectionresults_path)),
                             lambda row: row[2]):

        # Replace special characters in categories
        if str(key) != 'category':
            key = str(key).replace("/","-")
            key = str(key).replace(".", "")
            key = str(key).replace("(", "")
            key = str(key).replace(")", "")

            key = key + '.txt'
            detectionresults_single_path = xways_folder / key

            with open(str(detectionresults_single_path), 'a') as rf:

                for row in rows:
                    rf.write(row[0] + "\n")
                rf.flush()

    # Get a list of all files in results directory
    resultsfiles = os.listdir(str(xways_folder))

    # Prepend them with MD5 for seamless import into XWays
    for file in resultsfiles:
        line = "md5"
        if file[-3:] == 'txt' and file != 'Logfile.txt':
            with open(str(xways_folder / file), 'r+') as ff:
                content = ff.read()
                ff.seek(0,0)
                ff.write(line.rstrip('\r\n') + '\n' + content)


######
#
# Main program function
# First initiates required parameters and variables, then loads the GUI
# After which the image and video load functions are triggered based on the input parameters
# Finally, the detection is executed and results written to the place requested
#
######

# Prevent execution when externally called
if __name__ == '__main__':

    startTime = datetime.now()

    ######
    # Collecting parameters via GUI
    ######

    sg.ChangeLookAndFeel('Dark')

    layout = [[sg.Text('General Settings', font=("Helvetica", 13), text_color='sea green')],
              [sg.Text('Please specify the folder holding the media data:')],
              [sg.Input(), sg.FolderBrowse('Browse', initial_folder=Path.home(), button_color=('black', 'grey'))],
              [sg.Text('Where shall I place the results?')],
              [sg.Input(), sg.FolderBrowse('Browse', initial_folder=Path.home(), button_color=('black', 'grey'))],
              [sg.Text('Which things do you want to detect?')],
              [sg.Checkbox('Objects/Persons', size=(15, 2)),
               sg.Checkbox('Actions'),
               sg.Checkbox('IS Logos'),
               sg.Checkbox('Faces')],
              [sg.Text('Output Format:'), sg.Listbox(values=('Nuix', 'XWays', 'csv'), size=(29, 3))],
              [sg.Text('Video Settings', font=("Helvetica", 13), text_color='sea green')],
              [sg.Text('# of frames to be analyzed per Minute:', size=(36, 0))],
              [sg.Slider(range=(1, 120), orientation='h', size=(29, 20), default_value=30)],
              [sg.Text('Max. # of frames to be analyzed per Video:', size=(36, 0))],
              [sg.Slider(range=(1, 500), orientation='h', size=(29, 20), default_value=100)],
              [sg.Text('Check for & discard similar frames?'),
               sg.InputCombo(('Yes', 'No'), default_value='No', size=(10, 2))],
              [sg.OK(button_color=('black', 'sea green')), sg.Cancel(button_color=('black', 'grey'))]]

    layout_progress = [[sg.Text('Detection in progress')],
                       [sg.ProgressBar(10, orientation='h', size=(20, 20), key='progressbar')],
                       [sg.Cancel()]]

    # Render the GUI
    gui_input = sg.Window('BKP Media Detector').Layout(layout).Read()

    error = False

    for element in gui_input[1]:
        if element == '':
            error = True

    if error == True:
        sg.Popup('You have not populated all fields. Aborting!', title='Error', button_color=('black', 'red'), background_color=('grey'))
        exit()

    # Initiating progress meter
    if sg.OneLineProgressMeter('BKP Media Detector', 1, 9, 'key', 'Initializing variables & parameters...', orientation='h', size=(100, 10)) == False:
        exit()

    # Variable to determine minimum GPU Processor requirement & to disable TF log output
    # os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Validating TF version
    if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

    # Defining multiple needed variables based on GUI input & adding object_detection directory to path
    PATH_TO_INPUT = Path(gui_input[1][0])
    TEST_IMAGE_PATHS = Path.iterdir(PATH_TO_INPUT)
    number_of_input = 0
    for elements in Path.iterdir(PATH_TO_INPUT):
        number_of_input += 1
    PATH_TO_RESULTS = Path(gui_input[1][1])
    PATH_TO_OBJECT_DETECTION_DIR = '/home/b/Programs/tensorflow/models/research'  # PLACEHOLDER-tobereplacedWithPathtoDirectory
    sys.path.append(PATH_TO_OBJECT_DETECTION_DIR)
    REPORT_FORMAT = gui_input[1][6]
    frames_per_second = gui_input[1][7] / 60
    max_frames_per_video = gui_input[1][8]
    video_sensitivity_text = gui_input[1][9]
    if video_sensitivity_text == "Yes":
        video_sensitivity = 20
    else:
        video_sensitivity = 0

    # Check which models to apply and load their corresponding label maps
    from object_detection.utils import label_map_util

    graphlist = []
    indexlist = []

    MODEL1 = bool(gui_input[1][2])
    if MODEL1:
        OPEN_IMAGES_GRAPH = str(Path('Models/OpenImages/openimages.pb'))
        OPEN_IMAGES_LABELS = str(OPEN_IMAGES_GRAPH)[:-3] + '.pbtxt'
        OPEN_IMAGES_INDEX = label_map_util.create_category_index_from_labelmap(OPEN_IMAGES_LABELS)
        graphlist.append(OPEN_IMAGES_GRAPH)
        indexlist.append(OPEN_IMAGES_INDEX)

    MODEL2 = bool(gui_input[1][3])
    if MODEL2:
        AVA_GRAPH = str(Path('Models/AVA/ava.pb'))
        AVA_LABELS = str(AVA_GRAPH)[:-3] + '.pbtxt'
        AVA_INDEX = label_map_util.create_category_index_from_labelmap(AVA_LABELS)
        graphlist.append(AVA_GRAPH)
        indexlist.append(AVA_INDEX)

    MODEL3 = bool(gui_input[1][4])
    if MODEL3:
        SPECIAL_DETECTOR_GRAPH = str(Path('Models/ISLogos/islogos.pb'))
        SPECIAL_DETECTOR_LABELS = str(SPECIAL_DETECTOR_GRAPH)[:-3] + '.pbtxt'
        SPECIAL_DETECTOR_INDEX = label_map_util.create_category_index_from_labelmap(SPECIAL_DETECTOR_LABELS)
        graphlist.append(SPECIAL_DETECTOR_GRAPH)
        indexlist.append(SPECIAL_DETECTOR_INDEX)

    FACE_MODEL = bool(gui_input[1][5])

    # Update the progress indicator
    if sg.OneLineProgressMeter('BKP Media Detector', 2, 9, 'key', 'Process started. Loading images...', orientation='h', size=(100, 10)) == False:
        exit()

    # Create logfile
    logfile = open(str(PATH_TO_RESULTS / 'Logfile.txt'), 'w')
    logfile.write('***DETECTION LOG***\n')
    logfile.write("*" + str(datetime.now()) + ': Process started. Loading images...*\n')

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
    if sg.OneLineProgressMeter('BKP Media Detector', 3, 9, 'key', 'Loading Videos...',orientation='h',size=(100, 10)) == False: exit()

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
            number_of_videos += 1

    for error in errors:
        logfile.write(error)
    logfile.flush()

    # Split the result from the loading function into hashes and image arrays
    hashvalues, image_nps = zip(*final_images)

    # Update the progress indicator & logfile
    if sg.OneLineProgressMeter('BKP Media Detector', 4, 9, 'key', 'Starting detection...',orientation='h',size=(100, 10)) == False: exit()
    logfile.write("*" + str(datetime.now()) + ": Loading completed. Detecting...*\n")

    # Execute detection
    detectedLogos, errorcount = run_inference_for_multiple_images(image_nps, hashvalues)

    # Check whether an Xways report needs to be created
    if REPORT_FORMAT[0] == 'XWays':
        createXWaysReport()

    # Write process statistics to logfile
    logfile.write("*Results: " + str(PATH_TO_RESULTS / 'Detection_Results.csv*\n'))
    logfile.write("*Total Amount of Files: " + str(number_of_input) + " (of which " + str(number_of_images + number_of_videos) + " where processed.)*\n")
    logfile.write("*Processed Images: " + str(number_of_images) + "*\n")
    logfile.write("*Processed Videos: " + str(number_of_videos) + " (analyzed " + str(frames_per_second * 60) + " frames per minute, up to max. 100) with a " + video_sensitivity_text + " sensitivity.*\n")
    logfile.write("*Applied models:\n")
    for y in range(0, len(graphlist)): logfile.write(graphlist[y] + "\n")
    if FACE_MODEL: logfile.write("Face Detector\n")
    logfile.write("*Processing time: " + str(datetime.now() - startTime) + "*\n")
    logfile.flush()
    logfile.close()

    # Update progress indicator
    sg.OneLineProgressMeter('BKP Media Detector', 9, 9, 'key', 'Detection finished',orientation='h',size=(100, 10))

    # Deliver final success pop up to user
    sg.Popup('The detection was successful',
             'The results are placed here:',
             'Path: "{}"'.format(str(PATH_TO_RESULTS)))


