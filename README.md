YOLOv4 Object Detection using Darknet (Google Colab)
===================================================

This project demonstrates how to run YOLOv4 Object Detection using the Darknet
framework on Google Colab. It includes steps for compiling Darknet with GPU,
cuDNN, and OpenCV support, performing detection on images, and running inference
on videos.

---------------------------------------------------
FEATURES
---------------------------------------------------
- Compile Darknet with GPU, cuDNN, and OpenCV
- Run YOLOv4 on images
- Run YOLOv4 on video files
- Display detection results in the notebook
- Fully reproducible setup for Google Colab

---------------------------------------------------
REQUIREMENTS
---------------------------------------------------
- Google account
- Google Colab with GPU runtime enabled
  (Runtime → Change runtime type → GPU)

No local installation required.

---------------------------------------------------
INSTALLATION & SETUP
---------------------------------------------------

1. Clone Darknet:
    !git clone https://github.com/AlexeyAB/darknet
    %cd darknet/

2. Enable GPU, cuDNN, cuDNN_HALF, and OpenCV:
    !sed -i 's/GPU=0/GPU=1/' Makefile
    !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    !sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    !sed -i 's/OPENCV=0/OPENCV=1/' Makefile

3. Compile Darknet:
    !make -j8

4. Download YOLOv4 weights:
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

---------------------------------------------------
RUN YOLOv4 ON AN IMAGE
---------------------------------------------------

Run detection:
    !./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg

Display the result in Colab:
    import cv2
    import matplotlib.pyplot as plt

    def show_detection(path):
        image = cv2.imread(path)
        plt.figure(figsize=(18,10))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

---------------------------------------------------
RUN YOLOv4 ON VIDEO
---------------------------------------------------

Command:
    !./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights \
    -dont_show /content/drive/MyDrive/Videos/video_street.mp4 \
    -i 0 \
    -out_filename /content/drive/MyDrive/Videos/video_street_result1.avi

---------------------------------------------------
GOOGLE DRIVE INTEGRATION (optional)
---------------------------------------------------

    from google.colab import drive
    drive.mount('/content/drive')

---------------------------------------------------
GPU CHECK
---------------------------------------------------

    import tensorflow as tf
    tf.test.gpu_device_name()

---------------------------------------------------
OUTPUT
---------------------------------------------------
- Detection results on images
- Output video with bounding boxes
- Results saved to Google Drive

---------------------------------------------------
PROJECT STRUCTURE
---------------------------------------------------

    ├── Object_Detection.ipynb
    ├── README.txt
    ├── images/
    └── results/


---------------------------------------------------
LICENSE
---------------------------------------------------
Uses the Darknet YOLOv4 license from the AlexeyAB repository.
