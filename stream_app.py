# import streamlit as st
# #st.write("Hello ,let's learn how to build a streamlit app together")
#
#
# st.image(r"uploads\detectedcrop_lp1.png")


#streamlit run C:\Users\osaid\pycharm projects\FlaskALPR\stream_app.py
#streamlit run stream_app.py

# st.title ("this is the app title")
# st.header("this is the markdown")
# st.markdown("this is the header")
# st.subheader("this is the subheader")
# st.caption("this is the caption")
# st.code("x=2021")
# st.latex(r''' a+a r^1+a r^2+a r^3 ''')

#st.image("uploads/car_lp.jpg")
# st.audio("Audio.mp3")
# st.video("video.mp4")
#st.file_uploader('upload your car image')

# st.checkbox('yes')
# st.button('Click')
# st.radio('Pick your gender',['Male','Female'])
# st.selectbox('Pick your gender',['Male','Female'])
# st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
# st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
# st.slider('Pick a number', 0,50)

# st.number_input('Pick a number', 0,10)
# st.text_input('Email address')
# st.date_input('Travelling date')
# st.time_input('School time')
# st.text_area('Description')
# st.file_uploader('Upload a photo')
# st.color_picker('Choose your favorite color')

#----------------------------------------------------------------------------

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import easyocr
from ultralytics import YOLO
import io
import torch
import torchvision.transforms as transforms
import tempfile
import os
import pandas as pd
#---------logo and title set--------------
top_image = Image.open('static/logo1.PNG')
# bottom_image = Image.open('static/banner_bottom.png')
# main_image = Image.open('static/background.jpg')



#st.image(main_image,use_column_width='auto')
# st.title(' ALPR & VMMR Surveillance System VSL Lab 🚘🚙')
# st.sidebar.image(top_image, use_column_width='auto')
#st.sidebar.header('Input 🛠')
#selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image", "Live Video Feed"])
#st.sidebar.image(bottom_image,use_column_width='auto')



#---------logo and title set--------------
#-------
# Load the pretrained model
model_make = torch.load("mohsind_res34.pt", map_location=torch.device('cpu'))
model_make.eval()

# Define class labels (adjust these according to your LP_dataset)
class_labels = ['Daiatsu_Core', 'Daiatsu_Hijet', 'Daiatsu_Mira', 'FAW_V2', 'FAW_XPV', 'Honda_BRV', 'Honda_city_1994', 'Honda_city_2000', 'Honda_City_aspire', 'Honda_civic_1994', 'Honda_civic_2005', 'Honda_civic_2007', 'Honda_civic_2015', 'Honda_civic_2018', 'Honda_Grace', 'Honda_Vezell', 'KIA_Sportage', 'Suzuki_alto_2007', 'Suzuki_alto_2019', 'Suzuki_alto_japan_2010', 'Suzuki_carry', 'Suzuki_cultus_2018', 'Suzuki_cultus_2019', 'Suzuki_Every', 'Suzuki_highroof', 'Suzuki_kyber', 'Suzuki_liana', 'Suzuki_margala', 'Suzuki_Mehran', 'Suzuki_swift', 'Suzuki_wagonR_2015', 'Toyota HIACE 2000', 'Toyota_Aqua', 'Toyota_axio', 'Toyota_corolla_2000', 'Toyota_corolla_2007', 'Toyota_corolla_2011', 'Toyota_corolla_2016', 'Toyota_fortuner', 'Toyota_Hiace_2012', 'Toyota_Landcruser', 'Toyota_Passo', 'Toyota_pirus', 'Toyota_Prado', 'Toyota_premio', 'Toyota_Vigo', 'Toyota_Vitz', 'Toyota_Vitz_2010']


# Define image preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image_make_model(image_bytes):
    image1 = Image.open(image_bytes)
    image_tensor = preprocess(image1).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_labels[predicted_idx.item()]

    return predicted_class

#-------

UPLOAD_FOLDER = "static/"

# Load the YOLO model
model = YOLO('best.pt')

reader = easyocr.Reader(lang_list=['en'])

st.sidebar.image(top_image, use_column_width='auto')
# Create a sidebar with three options
st.sidebar.title("Choose an option")
option = st.sidebar.radio("", ["Detect with Image", "Detect with Video", "Detect with IP Camera Live", "Detect Make & Model"])

# Main app content
#st.title("Number Plate Detection")  ---osaid** old title
st.title(' ALPR & VMMR Surveillance System VSL Lab 🚘🚙')


makemodel = []
colorm = []

# Function for number plate detection
def detect_number_plate(image):
    # Add your number plate detection code here
    # Return the image with the number plate highlighted

    # Sample code for demonstration (replace this with your own logic)
    plate_image = image.copy()
    cv2.rectangle(plate_image, (100, 100), (300, 200), (0, 255, 0), 2)

    return plate_image
#-----make fucntion which return number plate image..***

global frame_counter


def detect_number_plate_from_image(image):
    frame_counter = 0

    results = model(image)

    # Extract license plate region and text using EasyOCR
    detections_ = []
    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        detections_.append([x1, y1, x2, y2, score])

    if detections_:
        # Crop and save the detected frame as a separate image
        #im = Image.fromarray(image[..., ::-1])
        cropped_frame = image.crop(detections_[0][0:4])

        frame_counter += 1
        image_name = f"detected_frame_{frame_counter}.png"
        cropped_frame.save(UPLOAD_FOLDER + image_name)
        img = UPLOAD_FOLDER + image_name
    return img


# Function to extract text from an image using EasyOCR
def extract_text(image):
    # reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    return result

def detect_number_plate_from_video(video):
    pass

def detect_number_plate_from_ipcamera(ipaddress):
    pass

def detect_color(imagec):

    from color_recognition_api import color_histogram_feature_extraction
    from color_recognition_api import knn_classifier

    prediction = 'n.a.'

    # checking whether the training data is ready
    PATH = './training.data'

    color_histogram_feature_extraction.color_histogram_of_test_image(imagec)
    prediction = knn_classifier.main('training.data', 'test.data')
    return prediction


def read_frames(video):
    cap = cv2.VideoCapture(video)
    frame_counter = 0
    #frame_count = 0
    while True:
        success, frame = cap.read()
        #frame_count += 1
        #if frame_count % 4 == 0:
        if success == True:
            # Run YOLOv5 inference on the frame
            results = model(frame)

            # Extract license plate region and text using EasyOCR
            detections_ = []
            for detection in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                detections_.append([x1, y1, x2, y2, score])

            if detections_:
                # Crop and save the detected frame as a separate image
                im = Image.fromarray(frame[..., ::-1])
                cropped_frame = im.crop(detections_[0][0:4])

                frame_counter += 1
                image_name = f"detected_frame_{frame_counter}.png"
                cropped_frame.save(UPLOAD_FOLDER + image_name)
                img = UPLOAD_FOLDER + image_name


            # Convert annotated frame back to NumPy array
            #annotated_frame = results.render()[0].numpy()
            annotated_frame = results[0].plot()

            #st.image(annotated_frame)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            #print(frame_bytes)  #osaid
            #st.image(frame_bytes)

            yield frame_bytes , img
#-----------------------------------------


# Handle different options
if option == "Detect with Image":
    st.title("Number Plate Detection with Image")
    uploaded_image = st.file_uploader("Upload an image for number plate detection", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Number Plate"):
            #---write your function name here-****
            detected_image = detect_number_plate_from_image(image)
            #detected_image = detect_number_plate(np.array(image))

            st.image(detected_image, caption="Detected Number Plate", use_column_width=True)

            text_extraction_result = extract_text(detected_image)
            st.write("Extracted Text:")
            for text in text_extraction_result:
                st.write(text[1])

#detection from video..
elif option == "Detect with Video":
    st.title("Number Plate Detection From Video")
    uploaded_video = st.file_uploader("Upload a video for number plate detection", type=["mp4"])


    if uploaded_video:
        video_file_buffer = uploaded_video.read()
        st.video(video_file_buffer)

        if st.button("Detect Number Plate"):
            #print(f'uploaded video type name{uploaded_video.name}')

            # Add code to process the video and detect number plates frame by frame
            # Display the processed video with number plates highlighted

            FRAME_WINDOW = st.image([])
            detected_plate = st.image([])
            detected_text = st.text([])

            caption = st.caption([])
            head = st.header([])

            for frame_bytes, img in read_frames(uploaded_video.name):
                FRAME_WINDOW.image(frame_bytes)
                lp_text = reader.readtext(img)

                detected_plate.image(img, caption='Detecrted Car plate')

                # caption.caption('detected Car text')
                head.header('detected Car text')

                detected_text.write(lp_text[0][1])




elif option == "Detect with IP Camera Live":
    st.title("Number Plate Detection From Live Camera")
    camera_ip = st.text_input("Enter the IP Camera URL")

    if camera_ip:
        st.write(f"Live feed from IP Camera: {camera_ip}")

        # Add code to display the live video feed from the IP camera
        # Perform number plate detection and text extraction in real-time
        pass

#---detect make and model-----------
elif option == "Detect Make & Model":
    st.title("Vehicle Make, Model & Color Detection From Image")
    temp_dir = tempfile.TemporaryDirectory()

    uploaded_image = st.file_uploader("Upload an image for Make & Model Detection", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        file_path = os.path.join(temp_dir.name, uploaded_image.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_image.read())

        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        readimage = cv2.imread(file_path)
        vehicle_color = detect_color(readimage)

        if st.button("Detect Make & Model"):
            image_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model_make(image_tensor)
                # print('predicted outputs--------', outputs)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_class = class_labels[predicted_idx.item()]

            # st.write(f"Predicted Make and Model: {predicted_class}")  --can use
            # st.write(f"predicted color of car {vehicle_color}")  --can use

            #vehicle_color = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)

            makemodel.append(predicted_class)
            colorm.append(vehicle_color)

            data = {
                "Car Make/Model": makemodel,
                "Car Color": colorm
            }
            df = pd.DataFrame(data)

            # Display the DataFrame
            st.write("Detected Make Modela and Color:")
            st.dataframe(df)

# Add any other customization or components you need


#---------------------
# import cv2
# import time
# import streamlit as st
#
# ### CONNECT TO CAMERA
# # either like this using your own camera IP
# #capture = cv2.VideoCapture('rtsp://192.168.1.64/1')
# # or like this  'rtsp://admin:asdfg123*@172.16.38.73/1'
# capture = cv2.VideoCapture('carvideo1.mp4')
#
# ### Check box to turn on camera
# run = st.checkbox("Turn on camera",value=False)
#
# ### MAKE PLACE HOLDER FOR VIDEO FRAMES
# FRAME_WINDOW =st.image([])
#
# ### GRAB NEW IMAGE
# if run:
#     while True:
#         x, frame = capture.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)
#     #time.sleep(0.025)
