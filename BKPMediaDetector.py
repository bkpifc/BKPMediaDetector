#!/usr/bin/env python3

######
# General Detector
# 06.12.2018 / Last Update: 20.05.2021
# LRB
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
import face_recognition
import subprocess
from itertools import groupby
from distutils.version import StrictVersion
from PIL import Image
from datetime import datetime
from time import strftime
from time import gmtime
from multiprocessing import Pool
from Models.Face import detect_face
from pathlib import Path
from openvino.inference_engine import IENetwork, IECore
from AudioAnalysis import audioAnalysis

######
# Worker function to check the input provided via the GUI
#######
def validateInput(gui_input):

    error = False

    #Validate input
    # for element in gui_input[1][0:7]:
    #     if element == '' or []:
    #         error = True

    if gui_input[0] == "Cancel" or len(gui_input[1][8]) == 0:
        error = True

    if bool(gui_input[1][5]) == True and gui_input[1][12] == "":
        error = True

    if error == True:
        sg.Popup('You have not populated all required fields. Aborting!', title='Error', button_color=('black', 'red'), background_color=('grey'))
        exit()

######
# Worker function to update the progress bar
######
def updateProgressMeter(step, customText):
    if sg.OneLineProgressMeter('BKP Media Detector', step, 12, 'key', customText, orientation='h', size=(50, 25)) == False:
        exit()

######
# Worker function to prepare and reshape the input images into a Numpy array
# and to calculate the MD5 hashes of them.
######
def load_image_into_numpy_array(image_path):

    try:
        image_path = str(image_path)

        # Open, measure and convert image to RGB channels
        image = Image.open(image_path)
        (im_width, im_height) = image.size

        if int(im_width) < 34 or int(im_height) < 34:
            logfile.write("Insufficient file dimensions: " + str(image_path) + "\n")
            return None

        if int(im_width) > 4512 or int(im_height) > 3008:
            maxheight = int(3008)
            maxwidth = int(4512)
            resize_ratio = min(maxwidth/im_width, maxheight/im_height)
            im_width = int(im_width * resize_ratio)
            im_height = int(im_height * resize_ratio)
            image = image.resize((im_width, im_height))

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

        return image_path, hashvalue, np_array

    #Throw errors to stdout
    except IOError or OSError:
        magictype = str(magic.from_file((image_path), mime=True))
        # If image file cannot be read, check if it is a video
        if magictype[:5] == 'video': #or magictype[12:17] == 'octet':
            # If so, return a video flag instead of numpy array
            flag = "VIDEO"
        elif magictype[:5] == 'audio':
            flag = "AUDIO"
        elif magictype[12:17] == 'octet':
            flag = "OCTET"
        else:
            image_path = "Could not open file: " + str(image_path) + " (" + str(magictype) + ")\n"
            flag = "ERROR"
        return image_path, flag

    except:
        magictype = str(magic.from_file((image_path), mime=True))
        logfile.write("General error with file: " + str(image_path) + " (" + str(magictype) + ")\n")


def check_video_orientation(image_path):

    # Function to check video rotation with ffprobe and return corresponding CV2 rotation code
    try:
        cmnd = ['ffprobe', '-loglevel', 'error', '-select_streams', 'v:0', '-show_entries', 'stream_tags=rotate', '-of',
                'default=nw=1:nk=1', image_path]
        p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        orientation = out.decode('utf-8')

        if orientation == '':
            rotation = 3
        elif int(orientation) == 180:
            rotation = 1
        elif int(orientation) == 90:
            rotation = 0
        else:
            rotation = 2

        return rotation

    except:
        logfile.write("Cannot determine video rotation: " + str(image_path) + "\n")

######
# Worker function to prepare and reshape the input videos to a Numpy array
# and to calculate the MD5 hashes of them.
# The function analyzes as much frames as indicated in the variable "frames_per_second" (Default = 0.5)
######
def load_video_into_numpy_array(image_path):

    videoframes = []
    old_hash = None
    # Loading the video via the OpenCV framework
    try:
        rotation = check_video_orientation(image_path)
        vidcap = cv2.VideoCapture(image_path)
        im_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Switch height/width if video is to be rotated 90/270 degrees
        if rotation == 0 or rotation == 2:
            im_width_new = im_height
            im_height_new = im_width
            im_width = im_width_new
            im_height = im_height_new

        # Calculating frames per second, total frame count and analyze rate
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        framecount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        analyze_rate = int(framecount / fps * frames_per_second)

        if 0 < analyze_rate < max_frames_per_video:
            int(analyze_rate)
        elif analyze_rate >= int(max_frames_per_video):
            analyze_rate = int(max_frames_per_video) #Limiting maximum frames per video
        else:
            videoerror = 'Unable to extract frames from video: ' + str(image_path) + '\n'
            return videoerror

        # Hashing the video once
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
             for chunk in iter(lambda: f.read(4096), b""):
                 hash_md5.update(chunk)
        hashvalue = hash_md5.hexdigest()

        # Extracting the frames from the video
        for percentile in range(0, analyze_rate):

            vidcap.set(cv2.CAP_PROP_POS_FRAMES, (framecount / analyze_rate) * percentile)
            success, extracted_frame = vidcap.read()
            if rotation != 3:
                extracted_frame = cv2.rotate(extracted_frame, rotation)

            extracted_frame = cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2RGB)
            timecode = ((framecount / analyze_rate) * percentile) / fps
            timecode = str(strftime("%H:%M:%S", gmtime(timecode)))

            # And reshape them into a numpy array
            np_array = np.array(extracted_frame).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

            if video_sensitivity > 0:
                # Compare the frame with the previous one for similarity, and drop if similar
                frame_to_check = Image.fromarray(np_array)
                new_hash = imagehash.phash(frame_to_check)
                if old_hash is None or (new_hash - old_hash > video_sensitivity):
                    cluster = str(image_path + ";" + str(timecode)), hashvalue, np_array
                    videoframes.append(cluster)
                    old_hash = new_hash
            else:
                cluster = str(image_path + ";" + str(timecode)), hashvalue, np_array
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
# Detection within loaded images with Tensorflow framework
# Creation of output file with hashes, detection scores and class
######
def run_inference_for_multiple_images(image_paths, images, hashvalues):

    # Open the results file again
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'a')

    for y in range(0, len(graphlist)):
        # Create TF Session with loaded graph
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            logfile.write("*" + str(datetime.now()) + ": \tStarting detection with model " + str(y + 1) + " of " + str(len(graphlist)) + "*\n")

            # Update progress indicator
            updateProgressMeter(7 + y, 'Detecting with model {}'.format(graphlist[y]))

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
                    updateProgressMeter(7 + y, str(graphlist[y]) + '\nFile ' + str(index) + ' of ' + str(len(images)))
                    try:
                        output_dict = sess.run(tensor_dict,
                                               feed_dict={image_tensor: np.expand_dims(image, 0)})

                        # all outputs are float32 numpy arrays, so convert types as appropriate
                        output_dict['num_detections'] = int(output_dict['num_detections'][0])
                        output_dict['detection_scores'] = output_dict['detection_scores'][0]
                        detectionhit = output_dict['num_detections']
                        output_dict['detection_classes'] = output_dict['detection_classes'][0]

                        hashvalue = hashvalues[index]
                        image_path = image_paths[index]

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
                                    line = ",".join([Path(image_path).name, hashvalue, scorestring, category['name']])
                                detectionresults.write(line + "\n")

                    except tf.errors.InvalidArgumentError:

                        logfile.write("Unable to process file dimensions of file with hash: \t" + str(hashvalue) + "\n")

                logfile.write("*" + str(datetime.now()) + ": \tFinished detection with model " + str(y + 1) + "*\n")

    detectionresults.flush()
    detectionresults.close()

######
# Detect and count faces in loaded images
# Prepare and call age/gender detection once done
######
def faceDetection(image_paths, images, hashvalues):

    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'a')

    # Updating progress bar and logfile
    updateProgressMeter(10, 'Detecting with Face/Age/Gender Detector')
    logfile.write("*" + str(datetime.now()) + ": \tStarting detection with face/age/gender detection model*\n")

    # Applying constants as defined in Facenet
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    # Creating different TF Session
    with tf.Session() as sess:

        # read pnet, rnet, onet models from Models/Face directory
        facemodel_path = Path('Models/Face')
        pnet, rnet, onet = detect_face.create_mtcnn(sess, str(facemodel_path))

        # Helperlists for age/gender detection
        facelist = []
        imagelist = []

        # Inference for all images
        for index, image in enumerate(images):
            updateProgressMeter(10, 'Detecting with Face/Age/Gender Detector' + '\nFile ' + str(index) + ' of ' + str(len(images)))
            try:
                bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

                # If a face was detected, go on
                if nrof_faces > 0:
                    detectedFaces = bounding_boxes[:, 0:4]
                    detectedFacesArray = []
                    img_size = np.asarray(image.shape)[0:2]

                    if nrof_faces > 1:
                        for single_face in range(nrof_faces):
                            detectedFacesArray.append(np.squeeze(detectedFaces[single_face]))
                    else:
                        detectedFacesArray.append(np.squeeze(detectedFaces))

                    # Crop the detected face and add it to the list to conduct age/gender identification
                    for x, detectedFaces in enumerate(detectedFacesArray):
                        detectedFaces = np.squeeze(detectedFaces)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(detectedFaces[0], 0)
                        bb[1] = np.maximum(detectedFaces[1], 0)
                        bb[2] = np.minimum(detectedFaces[2], img_size[1])
                        bb[3] = np.minimum(detectedFaces[3], img_size[0])
                        cropped_Face = image[bb[1]:bb[3], bb[0]:bb[2], :]
                        facelist.append(cropped_Face)
                        imagelist.append(index)

                # Write the results of the face detection into the resultsfile
                if not len(bounding_boxes) == 0:
                    hashvalue = hashvalues[index]
                    number_of_faces = len(bounding_boxes)
                    if REPORT_FORMAT[0] == 'Nuix':
                        line = "Face,md5:" + hashvalue
                    else:
                        line = str(Path(image_paths[index]).name) + "," + str(hashvalue) + ",FACES," + str(
                            number_of_faces) + "Faces"

                    detectionresults.write(line + "\n")

            except tf.errors.InvalidArgumentError:
                errorcount += 1
                logfile.write("Unable to detect faces in file with hash: \t" + str(hashvalue) + "\n")

        # Conduct age/gender recognition based on the list of detected & cropped faces
        if len(facelist) != 0:
            age_gender_detection(imagelist, facelist, hashvalues, image_paths)

    logfile.write("*" + str(datetime.now()) + ": \tFinished detection with face/age/gender detection model*\n")

    detectionresults.flush()
    detectionresults.close()

######
# Detection with the OPEN VINO Framework
# Evaluate Age & Gender based on input faces
######
def age_gender_detection(imagelist, facelist, hashvalues, image_paths):

    # Acquire the age-gender detection model
    model_path = Path('Models/OpenVINO/age-gender')
    model_xml = str(model_path / 'model.xml')
    model_bin = str(model_path / 'model.bin')

    # Reopen the results file
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'a')

    # Plugin initialization for specified device and load extensions library if specified
    ie = IECore()

    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    net.batch_size = len(facelist)

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))

    # Loading model to the plugin
    exec_net = ie.load_network(network=net, device_name='CPU')

    # Resize and reshape input faces
    for i in range(n):
        image = facelist[i]
        if image.shape[:-1] != (62, 62):
            h, w = image.shape[:2]

            # interpolation method
            if h > 62 or w > 62:  # shrinking image
                interp = cv2.INTER_AREA
            else:  # stretching image
                interp = cv2.INTER_CUBIC

            # aspect ratio of image
            aspect = w / h

            # compute scaling and pad sizing
            if aspect > 1:  # horizontal image
                new_w = 62
                new_h = np.round(new_w / aspect).astype(int)
                pad_vert = (62 - new_h) / 2
                pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
                pad_left, pad_right = 0, 0
            elif aspect < 1:  # vertical image
                new_h = 62
                new_w = np.round(new_h * aspect).astype(int)
                pad_horz = (62 - new_w) / 2
                pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
                pad_top, pad_bot = 0, 0
            else:  # square image
                new_h, new_w = 62, 62
                pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

            # set pad color
            padColor = 0
            if len(image.shape) is 3 and not isinstance(padColor, (
                    list, tuple, np.ndarray)):  # color image but only one color provided
                padColor = [padColor] * 3

            # scale and pad
            scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
            scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                            borderType=cv2.BORDER_CONSTANT, value=padColor)

            image = scaled_img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[i] = image

    # Conduct inference
    res = exec_net.infer(inputs={input_blob: images})

    # Process inference results
    for y in range(len(facelist)):
        probable_age = int(np.squeeze(res['age_conv3'][y]) * 100)
        if np.squeeze(res['prob'][y][0]) > 0.5:
            gender = "Female"
        else:
            gender = "Male"

        age_gender_combo = str(probable_age) + str(gender)

        # Write inference results to resultsfile
        hashvalue = hashvalues[imagelist[y]]
        if REPORT_FORMAT[0] == 'Nuix':
            line = str(age_gender_combo) + ",md5:" + hashvalue
        else:
            line = str(Path(image_paths[imagelist[y]]).name) + "," + str(hashvalue) + ",AGE-GENDER," + str(
                age_gender_combo)

        detectionresults.write(line + "\n")


######
# Detection with the OPEN VINO Framework
# Creation of output file with hashes, detection scores and class
######
def run_inference_openvino(image_paths, images, hashvalue):

    # Update progress meter and reopen results file
    updateProgressMeter(6, 'Detecting with OpenVINO Object Detector')
    logfile.write("*" + str(datetime.now()) + ": \tStarting detection with OpenVINO object detection model*\n")
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'a')

    # Fetch paths for openvino model
    model_path = Path('Models/OpenVINO/vgg19')
    model_xml = str(model_path / 'model.xml')
    model_bin = str(model_path / 'model.bin')
    model_labels = str(model_path / 'model.labels')
    temp_bilder = images

     # Plugin initialization for specified device and load extensions library if specified
    ie = IECore()

    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 4000

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))

    # Loading model to the plugin
    exec_net = ie.load_network(network=net, device_name='CPU')

    # Create batches to prevent RAM overload
    batches = tuple(temp_bilder[x:x + net.batch_size] for x in range(0, len(temp_bilder), net.batch_size))

    # Start sync inference
    for batch in batches:
        for index, temp_pic in enumerate(batch):
            temp_pic = cv2.resize(temp_pic, (w, h))
            temp_pic = temp_pic.transpose((2, 0, 1))
            images[index] = temp_pic

        res = exec_net.infer(inputs={input_blob: images})

        # Processing output blob
        res = res[out_blob]

        # Prepare label file
        with open(model_labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]

        # Clean inference results and write them to resultsfile
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)

            top_ind = np.argsort(probs)[-3:][::-1]

            for id in top_ind:
                if probs[id] >= 0.3:
                    # det_label = labels_map[id] if labels_map else "{}".format(id)
                    det_label = labels_map[id].split(sep=' ', maxsplit=1)[1]
                    if REPORT_FORMAT[0] == 'Nuix':
                        line = ",".join([det_label, "md5:" + hashvalue])
                    else:
                        line = ",".join([Path(image_paths[i]).name, hashvalue[i], str(probs[id]), str(det_label)])
                    detectionresults.write(line + "\n")

    logfile.write("*" + str(datetime.now()) + ": \tFinished detection with OpenVINO object detection model*\n")

######
# Worker function to load and encode known faces and to compare them against
# the provided input material
######
def faceRecognition(known_faces_path, image_paths, images, hashvalues):

    # Update progress bar
    updateProgressMeter(5, 'Conducting Face Recognition')

    known_face_counter = 0

    # Open the results file
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'a')

    OutputPictureFolder = PATH_TO_RESULTS / 'DetectedFaces'
    if not OutputPictureFolder.exists(): os.mkdir(str(OutputPictureFolder))

    # Initiate array to store known faces
    known_face_encodings = []
    known_face_names = []
    known_faces = Path.iterdir(Path(known_faces_path))

    # Create encodings and store them with names
    for known_face in known_faces:
        known_person_image = face_recognition.load_image_file(known_face)
        known_face_encodings.extend(face_recognition.face_encodings(known_person_image))
        known_face_names.append(Path(known_face).stem)

    logfile.write("*" + str(datetime.now()) + ": \tStarting face recognition with " + str(len(known_face_names)) + " known faces*\n")

    # Load images, detect faces, encode and compare them to the known faces
    for index, image_to_detect in enumerate(images):
        hashvalue = hashvalues[index]
        image_path = image_paths[index]
        updateProgressMeter(5, 'Face Reco Image ' + str(index) + ' of ' + str(len(images)))
        # Use GPU based model to detect & encode
        face_locations = face_recognition.face_locations(image_to_detect, model="cnn")
        face_encodings = face_recognition.face_encodings(image_to_detect, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=facereq_tolerance)
            name = "Unknown"

            # Check the face distance and get best match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # If there is a match, write it to the output file
            if name != "Unknown":
                known_face_counter += 1
                if REPORT_FORMAT[0] == 'Nuix':
                    line = ",".join([name, "md5:" + hashvalue])
                else:
                    line = ",".join([Path(image_path).name, hashvalue, "FACE-Match", name])
                detectionresults.write(line + "\n")

                if output_detFaces:
                    # Export detected face with bounding box
                    cv2.rectangle(image_to_detect, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(image_to_detect, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_to_detect, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                    savePath = str(OutputPictureFolder / str(Path(image_path).name)) + '.jpg'

                    detectedFace = Image.fromarray(image_to_detect)
                    detectedFace.save(savePath)

    logfile.write("*" + str(datetime.now()) + ": \tFace Recognition completed.*\n")

    detectionresults.flush()
    detectionresults.close()

    # Return amount of detected known faces
    return known_face_counter

######
# Worker function to conduct speech detection in audio files
# for all audio files detected
######
def audioSpeechDetection(audiolist):

    logfile.write("*" + str(datetime.now()) + ": \tStarting audio speech detection*\n")
    updateProgressMeter(11, 'Processing Audio Files')

    audiocounter = 0

    # Open the results file
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'a')

    pool = Pool(maxtasksperchild=100)
    result = pool.map(audioAnalysis.segmentSpeechDetection, audiolist, chunksize=10)
    pool.close()

    # Synchronize after completion
    pool.join()
    pool.terminate()

    result = [x for x in result if x != None]

    for processedAudio in result:
        speechPercentage, audiopath = processedAudio

        # Check for the video flag
        if not isinstance(speechPercentage, float):
            logfile.write("Unsupported audio file: " + str(audiopath) + "\n")
        else:
            speechPercentage, audiopath = processedAudio
            # Hashing the video once
            hash_md5 = hashlib.md5()
            with open(audiopath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            hashvalue = hash_md5.hexdigest()
            audiocounter += 1

            if REPORT_FORMAT[0] == 'Nuix':
                if speechPercentage != 0.0:
                    line = ",".join(["AUDIO-SPEECH", "md5:" + hashvalue])
            else:
                line = ",".join([Path(audiopath).name, hashvalue, str(speechPercentage), "AUDIO-SPEECH"])
            detectionresults.write(line + "\n")

    logfile.write("*" + str(datetime.now()) + ": \tAudio speech detection completed.*\n")

    detectionresults.flush()
    detectionresults.close()

    return audiocounter


######
# Split the report file to allow seamless integration into XWays Hash Database per category
######
def createXWaysReport():

    detectionresults_path = str(PATH_TO_RESULTS / 'Detection_Results.csv')
    xways_folder = PATH_TO_RESULTS / 'XWaysOutput'
    if not xways_folder.exists(): os.mkdir(str(xways_folder))

    for key, rows in groupby(csv.reader(open(detectionresults_path)),
                             lambda row: row[3]):

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
                    rf.write(row[1] + "\n")
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



    ######
    # Collecting parameters via GUI
    ######

    sg.ChangeLookAndFeel('Dark')

    layout = [[sg.Text('General Settings', font=("Helvetica", 13), text_color='sea green')],
              [sg.Text('Please specify the folder holding the media data:')],
              [sg.Input(), sg.FolderBrowse('Browse', initial_folder='/home/b/Desktop/TestBilder', button_color=('black', 'grey'))], #Path.home() = Initial folder
              [sg.Text('Where shall I place the results?')],
              [sg.Input(), sg.FolderBrowse('Browse', initial_folder='/home/b/Desktop/TestResults', button_color=('black', 'grey'))], #Path.home()
              [sg.Text('TENSORFLOW DETECTORS')],
              [sg.Checkbox('Objects/Persons', size=(15, 2)),
               sg.Checkbox('Actions'),
               sg.Checkbox('IS Logos'),
               sg.Checkbox("Face Recognition")],
              [sg.Text('OPEN VINO DETECTORS')],
              [sg.Checkbox('Objects-fast', size=(15, 2)),
               sg.Checkbox('Faces/Age/Gender')],
              [sg.Text('Output Format:'), sg.Listbox(values=('Nuix', 'XWays', 'csv'), size=(29, 3))],
              [sg.Text('Video Settings', font=("Helvetica", 13), text_color='sea green')],
              [sg.Text('# of frames to be analyzed per Minute:', size=(36, 0))],
              [sg.Slider(range=(1, 120), orientation='h', size=(29, 20), default_value=30)],
              [sg.Text('Max. # of frames to be analyzed per Video:', size=(36, 0))],
              [sg.Slider(range=(1, 500), orientation='h', size=(29, 20), default_value=100)],
              [sg.Text('Check for & discard similar frames?'),
               sg.InputCombo(('Yes', 'No'), default_value='No', size=(10, 2))],
              [sg.Text('Face Recognition', font=("Helvetica", 13), text_color='sea green')],
              [sg.Text('Specify folder with known faces (if FaceReq selected): ')],
              [sg.Input(), sg.FolderBrowse('Browse', initial_folder='/home/b/Desktop/known', button_color=('black', 'grey'))],
              [sg.Text('Specify face recognition tolerance (Default: 60%):', size=(48, 0))],
              [sg.Slider(range=(0, 100), orientation='h', size=(29, 20), default_value=60)],
              [sg.Checkbox('Output detected faces as jpg', size=(25, 2))],
              [sg.Text('Audio Settings', font=("Helvetica", 13), text_color='sea green')],
              [sg.Text('AUDIO PROCESSING')],
              [sg.Checkbox('Speech Detection', size=(15, 2))],
              [sg.OK(button_color=('black', 'sea green')), sg.Cancel(button_color=('black', 'grey'))]]

    layout_progress = [[sg.Text('Detection in progress')],
                       [sg.ProgressBar(12, orientation='h', size=(20, 20), key='progressbar')],
                       [sg.Cancel()]]

    # Render the GUI
    gui_input = sg.Window('BKP Media Detector').Layout(layout).Read()
    error = False

    # Validate input
    validateInput(gui_input)

    # Initiating progress meter
    updateProgressMeter(1, 'Initializing variables & parameters...')
    startTime = datetime.now()
    # Variable to determine minimum GPU Processor requirement & to disable TF log output
    # os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Validating TF version
    if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

    # Defining multiple needed variables based on GUI input & adding TF/OpenVINO directory to path
    PATH_TO_INPUT = Path(gui_input[1][0])
    TEST_IMAGE_PATHS = Path.iterdir(PATH_TO_INPUT)
    number_of_input = 0
    for elements in Path.iterdir(PATH_TO_INPUT):
        number_of_input += 1
    PATH_TO_RESULTS = Path(gui_input[1][1])
    PATH_TO_OBJECT_DETECTION_DIR = '/home/b/Programs/tensorflow/models/research'  # PLACEHOLDER-tobereplacedWithPathtoDirectory
    sys.path.append(PATH_TO_OBJECT_DETECTION_DIR)

    REPORT_FORMAT = gui_input[1][8]

    frames_per_second = gui_input[1][9] / 60
    max_frames_per_video = gui_input[1][10]
    video_sensitivity_text = gui_input[1][11]
    KNOWN_FACES_PATH = gui_input[1][12]
    facereq_tolerance = int(gui_input[1][13])/100
    output_detFaces = gui_input[1][14]

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

    FACE_RECOGNITION = bool(gui_input[1][5])
    OPEN_VINO_vgg19 = bool(gui_input[1][6])
    FACE_MODEL = bool(gui_input[1][7])
    AUDIO_SPEECH_DETECTION = bool(gui_input[1][15])

    # Update the progress indicator
    updateProgressMeter(2, 'Process started. Loading ' + str(number_of_input) + ' media files...')

    # Create logfile
    logfile = open(str(PATH_TO_RESULTS / 'Logfile.txt'), 'w')
    logfile.write('***DETECTION LOG***\n')
    logfile.write("*" + str(datetime.now()) + ': \tProcess started. Loading images...*\n')

    # Create resultsfile
    detectionresults_path = PATH_TO_RESULTS / 'Detection_Results.csv'
    detectionresults = open(str(detectionresults_path), 'w')
    if REPORT_FORMAT[0] == 'Nuix':
        detectionresults.write("tag,searchterm\n")
    else:
        detectionresults.write("name,hash,score,category\n")
    detectionresults.flush()
    detectionresults.close()

    # Initiate needed variables
    vidlist = []
    audiolist = []
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

    # Check for the different flags set by mimetype
    for processed_image in processed_images:
        if str(processed_image[1]) == "VIDEO":
            # If present, populate the video list
            vidlist.append(processed_image[0])
        elif str(processed_image[1]) == "AUDIO":
            audiolist.append(processed_image[0])
        elif str(processed_image[1]) == "OCTET":
            if processed_image[0][-3:] in ["mp4", "mov", "mpg", "avi", "exo", "mkv", "m4v", "ebm"]:
                vidlist.append(processed_image[0])
            else:
                audiolist.append(processed_image[0])
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
    updateProgressMeter(3, 'Loading ' + str(len(vidlist)) + ' Videos...')

    # Multiprocess the video load function on all CPU cores available
    pool = Pool(maxtasksperchild=10)
    videoframes = pool.map(load_video_into_numpy_array, vidlist, chunksize=2)
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
    if len(final_images) != 0:
        image_path, hashvalues, image_nps = zip(*final_images)

    # Update the progress indicator & logfile
    updateProgressMeter(4, 'Starting detection of ' + str(len(final_images)) + ' media files')
    logfile.write("*" + str(datetime.now()) + ": \tLoading completed. Detecting...*\n")

    # Conduct Face Recognition if needed
    if FACE_RECOGNITION:
        known_face_counter = faceRecognition(KNOWN_FACES_PATH, image_path, image_nps, hashvalues)

    # Conduct OpenVino VGG19 Model if needed
    if OPEN_VINO_vgg19:
        run_inference_openvino(image_path, image_nps, hashvalues)

    # Execute all other detection models
    if len(final_images) != 0:
        run_inference_for_multiple_images(image_path, image_nps, hashvalues)

    # Conduct face/age/gender detection
    if FACE_MODEL:
        faceDetection(image_path, image_nps, hashvalues)

    if AUDIO_SPEECH_DETECTION:
        audiofiles_processed = audioSpeechDetection(audiolist)
    else:
        audiofiles_processed = 0

    # Check whether an Xways report needs to be created
    if REPORT_FORMAT[0] == 'XWays':
        createXWaysReport()

    # Write process statistics to logfile
    logfile.write("*Results:\t\t\t" + str(PATH_TO_RESULTS / 'Detection_Results.csv*\n'))
    logfile.write("*Total Amount of Files:\t\t" + str(number_of_input) + " (of which " + str(number_of_images + number_of_videos + audiofiles_processed) + " were processed.)*\n")
    logfile.write("*Processed Images:\t\t" + str(number_of_images) + "*\n")
    logfile.write("*Processed Videos: \t\t" + str(number_of_videos) + " (analyzed " + str(frames_per_second * 60) + " frames per minute, up to max. 500) with the check for content-based duplicates set to " + video_sensitivity_text + "\n")
    logfile.write("*Processed Audio Files:\t\t" + str(audiofiles_processed) + "*\n")
    logfile.write("*Applied models:\n")
    for y in range(0, len(graphlist)): logfile.write("\t\t\t\t" + graphlist[y] + "\n")
    if OPEN_VINO_vgg19: logfile.write("\t\t\t\tOpenVINO Object Detector\n")
    if FACE_MODEL: logfile.write("\t\t\t\tFace-Age-Gender Detector\n")
    if FACE_RECOGNITION: logfile.write("\t\t\t\tFace Recognition (Known faces detected: " + str(known_face_counter) + ")\n")
    logfile.write("*Processing time:\t\t" + str(datetime.now() - startTime) + "*\n")
    logfile.write("*Time per processed file:\t" + str((datetime.now() - startTime) / (number_of_images + number_of_videos + audiofiles_processed)) + "*\n")
    logfile.flush()
    logfile.close()

    # Update progress indicator
    sg.OneLineProgressMeter('BKP Media Detector', 12, 12, 'key', 'Detection finished',orientation='h',size=(100, 10))

    # Deliver final success pop up to user
    sg.Popup('The detection was successful',
             'The results are placed here:',
             'Path: "{}"'.format(str(PATH_TO_RESULTS)))


