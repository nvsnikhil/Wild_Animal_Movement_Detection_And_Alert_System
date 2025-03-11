import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
from yolo_utils import infer_image, show_image

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
                        type=str,
                        default='./',
                        help='The directory where the model weights and configuration files are.')

    parser.add_argument('-w', '--weights',
                        type=str,
                        default='yolov5.weights',
                        help='Path to the file which contains the weights for YOLOv5.')

    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='yolov5.cfg',
                        help='Path to the configuration file for the YOLOv5 model.')

    parser.add_argument('-l', '--labels',
                        type=str,
                        default='yolov5.txt',
                        help='Path to the file having the labels in a new-line separated way.')

    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.5,
                        help='The model will reject boundaries with probability less than the confidence value.')

    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.3,
                        help='The threshold to use when applying Non-Max Suppression.')

    parser.add_argument('--download-model',
                        type=bool,
                        default=False,
                        help='Set to True if the model weights and configurations are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

    # Download the YOLO model if needed
    if FLAGS.download_model:
        subprocess.call(['./yolov5-coco/get_model.sh'])

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Initializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configuration for the YOLOv5 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Access ESP32-CAM stream
    stream_url = 'http://192.168.137.201/capture'  # Update with your ESP32-CAM stream URL
    print(f"[INFO] Connecting to ESP32-CAM stream at {stream_url}...")

    # Open the video stream
    vid = cv.VideoCapture(stream_url)

    if not vid.isOpened():
        print("[ERROR] Unable to access the stream. Check the IP and port.")
        exit()

    count = 0

    while True:
        # Read frame from the stream
        vid = cv.VideoCapture(stream_url)
        grabbed, frame = vid.read()
        if not grabbed:
            print("[INFO] Stream ended or connection lost.")
            break

        height, width = frame.shape[:2]

        if count == 0:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                    height, width, frame, colors, labels, FLAGS)
            count += 1
        else:
            frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                    height, width, frame, colors, labels, FLAGS,
                                                                    boxes, confidences, classids, idxs, infer=False)
            count = (count + 1) % 6

        # Display the output frame
        cv.imshow('ESP32-CAM Object Detection', frame)

        # Exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Cleaning up...")
    vid.release()
    cv.destroyAllWindows()
