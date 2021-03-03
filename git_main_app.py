import av
import cv2
import io
import logging
import logging.handlers
import numpy as np
import pandas as pd
import PIL
import queue
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import streamlit as st
import threading
import time
import torch
import urllib.request

from pathlib import Path
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from const import MODELS, IMAGES, CLASSES
from nlp_models import nlpNaiveBayes, nlpLogisticReg, nlpKNearestNeighb, nlpSVM, nlpKernelSVM, nlpDecisionTree, nlpRandomForest

def main():
    
    # Sidebar title
    st.sidebar.title('Explore my ML projects')

    # Dropdown menu
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome', 'Image Annotation', 'Object Detection', 'Natural Language Processing') #, 'Data Exploration', 'Other obj det')
    )

    if selected_box == 'Welcome':
        welcome()
    if selected_box == 'Image Annotation':
        annotation()
    if selected_box == 'Image Preprocessing':
        preprocess()
    if selected_box == 'Object Detection':
        object_detection()
    if selected_box == 'Natural Language Processing':
        nlp()

def welcome():
    st.header("Welcome")
    st.write("Hi, my name is Kellan. I'm a Machine Learning Scientist/Data Scientist. \
        My work primarily focuses on Computer Vision, specifically object detection and tracking. \
        I also have experience working with LiDAR and the creation and rendering of 3D point clouds.")
        
    st.write("This is a Streamlit app I've created and deployed using Heroku. \
        You can use the dropdown in the sidebar to explore some of my personal ML projects.")

def annotation():

    # Page header and description
    st.header("Image annotation for object detection")
    st.write("As part of the data collection step of any object detection pipeline, \
        we have to create the ground truth (GT) of each image. In this step, we draw \
        a bounding box around each object of interest in an image.")

    img_annotate = st.sidebar.file_uploader("Upload one image for annotation:", type=["png", "jpg"])
    
    # Specify canvas parameters in application
    stroke_color = "rgba(0,0,0,0)" #st.sidebar.color_picker("Stroke color hex: ")
    drawing_mode = "rect" #st.sidebar.selectbox(
    realtime_update = st.sidebar.checkbox("Update in real-time", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(50, 195, 203, 0.3)",  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color=stroke_color,
        background_color="",
        background_image=Image.open(img_annotate) if img_annotate else None,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        key="canvas",
    )

def preprocess():

    # Page header and description
    st.header("Pre-process images for object detection")
    st.write("There are many options for pre-processing your data. Some that \
        we use include converting our images to black and white (greyscale) \
        and doing mosaic augmentation. Click through below to see the \
        effect of each of these.")

    # Path to image
    image = '2020-05-14_12-47-56-8385525_colour.jpg'
    step = st.sidebar.radio("Which step of the pre-processing pipeline would you like to see?",
        ('Original', 'Greyscale', 'Mosaic Augmentation'))
    
    if step == 'Original':
        st.image(image, use_column_width=True)

    if step == 'Greyscale':
        # original = Image.open(image).convert('L')
        # original.save('pil-greyscale.png')
        st.image('2020-05-14_12-47-56-8385525.jpg', use_column_width=True)
    
    if step == 'Mosaic Augmentation':
        st.image('mosaic.png', use_column_width=True)

def object_detection():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {}

    # Page header and description
    st.header("YOLOv5 in (thermal) action")
    st.write("In this project, you'll find a trained YOLOv5 model (via PyTorch) to detect people and dogs in thermal images. \
        In the sidebar, you can choose the experiment and the confidence threshold you'd like to use for inferencing. \
        There are 6 different test images to run through the model, each showing different scenarios. \
        Pick your settings and press the 'Run' button.")
    
    with st.sidebar:
        # model = st.selectbox("Which model?", list(MODELS.keys()))

        # Show model variants if there are multiple model categories.
        if isinstance(MODELS[model], dict):  # different model variants
            model_variant = st.selectbox("Which experiment?", list(MODELS[model].keys()))
            inputs["model_func"] = MODELS[model][model_variant]
        else:  # only one variant
            inputs["model_func"] = MODELS[model]

        # # Show image choices
        # image_choice = st.radio("Which image?", list(IMAGES.keys()))
        # inputs["img_func"] = IMAGES[image_choice]

        # Show image variants if there are multiple image categories.
        if isinstance(IMAGES[model], dict):  # different image variants
            image_variant = st.selectbox("Which image?", list(IMAGES[model].keys()))
            inputs["img_func"] = IMAGES[model][image_variant]
        else:  # only one variant
            inputs["img_func"] = IMAGES[model]

    # Set model and image files
    model_file = inputs["model_func"]
    image_file = inputs["img_func"]
    image_file = Image.open(image_file)
    rgb_im = image_file.convert("RGB")

    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=model_file)    # custom model
    
    # Allow user to set confidence threshold
    model.conf = st.sidebar.slider(
        'Confidence threshold', value=0.75, min_value=0.0, max_value=1.0, step=.05)     # confidence threshold (0-1)

    imgs = [image_file]    # batched list of images

    st.sidebar.write("\n")
    detect_button = st.sidebar.button('Run')
    placeholder = st.empty()

    # Once user presses button show image with bounding boxes
    if detect_button:
        placeholder.empty()
        
        # Inference
        prediction = model(imgs, size=640)  # includes NMS

        # Plot
        for i, img in enumerate(imgs):
            # st.write('\nImage %g/%g: %s ' % (i + 1, len(imgs), img.shape), end='')
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            predictionResults = prediction.xyxy[i]
            if predictionResults is not None:  # is not None
                for *box, conf, cls in predictionResults:  # [xy1, xy2], confidence, class
                    # st.write('Class: %g' % cls, end='')
                    st.write('There is a ', CLASSES[int(cls.item())], 'in this image.')
                    st.write('\nConfidence: %.2f ' % (conf*100), end='')  # label
                    ImageDraw.Draw(img).rectangle(box, outline=(26, 147, 239), width=3)  # plot
            img.save('results%g.jpg' % i)  # save
            st.image('results%g.jpg' % i, use_column_width=True)
        
    else:
        st.image(image_file, use_column_width=True)

    st.write("\n \n \n \n")
    expander = st.beta_expander("Dataset info")
    expander.markdown("The images used to train this model can be found here: [Dogs & People Thermal Images](https://public.roboflow.com/object-detection/thermal-dogs-and-people/1). \
        Although slightly more challenging, it is straightforward to create your own dataset of images. \
        To do this you'd need to either scrape the web for images or gather your own, then use an image \
        annotation tool to create the ground truths (as seen on the Image Annotation page of this app). \
        Apps that I have used for this include [LabelImg](https://github.com/tzutalin/labelImg), [VoTT](https://github.com/microsoft/VoTT), [Roboflow](https://roboflow.com/)")

def nlp():
    st.header('NLP for restaurant reviews')
    st.write('ADD PROJECT DESCRIPTION')

    bayes = nlpNaiveBayes()
    lReg = nlpLogisticReg()
    knn = nlpKNearestNeighb()
    svm = nlpSVM()
    ksvm = nlpKernelSVM()
    decTree = nlpDecisionTree()
    randFor = nlpRandomForest()

    d = {
        'Algorithm': ['Naive Bayes', 'Logistic Regression', 'KNN', 'SVM', 'Kernel SVM', 'Decision Tree', 'Random Forest'],
        'Accuracy Score': [bayes[0], lReg[0], knn[0], svm[0], ksvm[0], decTree[0], randFor[0]],
        'Precision Score': [bayes[1], lReg[1], knn[1], svm[1], ksvm[1], decTree[1], randFor[1]],
        'Recall Score': [bayes[2], lReg[2], knn[2], svm[2], ksvm[2], decTree[2], randFor[2]]
    }

    df = pd.DataFrame(data=d)
    df

    st.write('ADD END RESULT/DECISION DESCRIPTION')

    st.write('ADD PREDICTION OF SINGLE REVIEW (whether pos/neg)')

    st.write("\n \n \n \n")
    expander = st.beta_expander("Score details and definitions")
    expander.markdown("Accuracy scores for each type\n \
        Precision Scores: the ability of classifier not to label a negative sample as positive. \n \
        Recall Scores: the ability of classifier find all the positive samples. \
    [Roboflow](https://roboflow.com/)")

if __name__ == "__main__":
    main()
