from PIL import Image, ImageOps, ImageDraw
import joblib
import easyocr
import requests

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# data manipulation
import pandas as pd
import numpy as np

# ML models
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

subject_classifier = joblib.load('subject_classifier.joblib')
section_classifier = joblib.load('section_classifier.joblib')

def extract_text(img, detail = 1):
    result = reader.readtext(img, detail = detail)
    return result


def draw_boxes(img, bounds, color="brown", width=2):
    draw = ImageDraw.Draw(img)
    for bound in bounds:
        p0, p1, p2, p3 = bound
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return img


def bounding_box_sorting(boxes):
    num_boxes = len(boxes)
    # sort from top to bottom and left to right
    sorted_boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)
    # print('::::::::::::::::::::::::::testing')
    # check if the next neighgour box x coordinates is greater then the current box x coordinates if not swap them.
    # repeat the swaping process to a threshold iteration and also select the threshold 
    threshold_value_y = 10
    for i in range(5):
      for i in range(num_boxes - 1):
          if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < threshold_value_y and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
              tmp = _boxes[i]
              _boxes[i] = _boxes[i + 1]
              _boxes[i + 1] = tmp
    return _boxes


st.title("Text Extraction App")
st.header("Math Question Detection")
st.text("Upload a question image to extract text")

reader = easyocr.Reader(['en'])

label_map = {'Maths': 0, 'Physics': 1, 'Chemistry': 2}


# using url of image
input_url = st.text_input("")
if len(input_url) != 0:
    st.write("")
    st.header("Input Image")
    im = Image.open(requests.get(input_url, stream=True).raw)
    st.image(im)
    st.header("Extracted Text")
    geometry_text_confidence = extract_text(input_url)
    my_dict = dict(zip(np.arange(len(geometry_text_confidence)),[i[0] for i in geometry_text_confidence]))
    sorted_geometry = [list(my_dict.keys())[list(my_dict.values()).index(i)] for i in bounding_box_sorting([i[0] for i in geometry_text_confidence])]
    result = [geometry_text_confidence[i] for i in sorted_geometry]
    g, t, c = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]
    st.write(t)
    #st.write([i[1] for i in geometry_text_confidence])
    st.write("")
    st.header("Detected Text")
    st.image(draw_boxes(im, g))
    st.write("")
    st.header("Detected Subject")
    subject = subject_classifier.predict([' '.join(t)])
    labels_map_subject = {0: 'Maths', 1: 'Physics', 2: 'Chemistry'}
    subject = list(map(labels_map_subject.get, subject))
    #st.write(list(label_map.keys())[list(label_map.values()).index(subject[0])])
    st.write(subject[0])
    st.header("Section Classifier")
    section = section_classifier.predict_proba([' '.join(t)])
    section = np.argmax(section, axis=1)
    labels_map_section = {0: 'aljabar', 1: 'analitik', 2: 'bilangan', 3: 'geometri', 4: 'kalkulus', 5: 'probabilitas', 6: 'statistika', 7: 'trigonometri'}
    section = list(map(labels_map_section.get, section))
    st.write(section[0])