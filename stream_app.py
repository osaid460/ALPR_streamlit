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

# import streamlit as st
# import time
# # Using object notation
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )
#
# # Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )
#
#
#
# with st.sidebar:
#     with st.echo():
#         st.write("This code will be printed to the sidebar.")
#
#     with st.spinner("Loading..."):
#         time.sleep(5)
#     st.success("Done!")
#------------------------------------------------------------------------------
# import streamlit as st
# import argparse
# from ultralytics import YOLO
# import io
# import os
# from PIL import Image
# import cv2
# import numpy as np
# from torchvision.models import detection
# import torch
# from torchvision import models
# from io import BytesIO
#
# st.set_page_config(
#     page_title="Auto NPR",
#     page_icon="✨",
#     layout="centered",
#     initial_sidebar_state="expanded",
# )
#
# @st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
# def instantiate_model():
#     model = YOLO('best.pt')
#     # model = torch.hub.load("ultralytics/yolov5", "custom", path = "model/last.pt", force_reload=True)
#     # model.eval()
#     # model.conf = 0.5
#     # model.iou = 0.45
#     return model
#
# @st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
# def download_success():
#     st.balloons()
#     st.success('✅ Download Successful !!')
#
# top_image = Image.open('static/banner_top.png')
# bottom_image = Image.open('static/banner_bottom.png')
# main_image = Image.open('static/main_banner.png')
#
# upload_path = "uploads/"
# download_path = "downloads/"
# model = instantiate_model()
#
# st.image(main_image,use_column_width='auto')
# st.title(' Automatic Number Plate Recognition 🚘🚙')
# st.sidebar.image(top_image,use_column_width='auto')
# st.sidebar.header('Input 🛠')
# selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image", "Live Video Feed"])
# st.sidebar.image(bottom_image,use_column_width='auto')
#
# if selected_type == "Upload Image":
#     st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
#     uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png","jpg","bmp","jpeg"])
#
#     if uploaded_file is not None:
#         with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
#             f.write((uploaded_file).getbuffer())
#         with st.spinner(f"Working... 💫"):
#             uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
#             downloaded_image = os.path.abspath(os.path.join(download_path,str("output_"+uploaded_file.name)))
#
#             with open(uploaded_image,'rb') as imge:
#                 img_bytes = imge.read()
#
#             img = Image.open(io.BytesIO(img_bytes))
#             results = model(img, size=640)
#             results.render()
#             for img in results.imgs:
#                 img_base64 = Image.fromarray(img)
#                 img_base64.save(downloaded_image, format="JPEG")
#
#             final_image = Image.open(downloaded_image)
#             print("Opening ",final_image)
#             st.markdown("---")
#             st.image(final_image, caption='This is how your final image looks like 😉')
#             with open(downloaded_image, "rb") as file:
#                 if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
#                     if st.download_button(
#                                             label="Download Output Image 📷",
#                                             data=file,
#                                             file_name=str("output_"+uploaded_file.name),
#                                             mime='image/jpg'
#                                          ):
#                         download_success()
#                 if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
#                     if st.download_button(
#                                             label="Download Output Image 📷",
#                                             data=file,
#                                             file_name=str("output_"+uploaded_file.name),
#                                             mime='image/jpeg'
#                                          ):
#                         download_success()
#
#                 if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
#                     if st.download_button(
#                                             label="Download Output Image 📷",
#                                             data=file,
#                                             file_name=str("output_"+uploaded_file.name),
#                                             mime='image/png'
#                                          ):
#                         download_success()
#
#                 if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
#                     if st.download_button(
#                                             label="Download Output Image 📷",
#                                             data=file,
#                                             file_name=str("output_"+uploaded_file.name),
#                                             mime='image/bmp'
#                                          ):
#                         download_success()
#     else:
#         st.warning('⚠ Please upload your Image 😯')
#
#
# else:
#     st.info('✨ The Live Feed from Web-Camera will take some time to load up 🎦')
#     live_feed = st.checkbox('Start Web-Camera ✅')
#     FRAME_WINDOW = st.image([])
#     cap = cv2.VideoCapture(0)
#
#     if live_feed:
#         while(cap.isOpened()):
#             success, frame = cap.read()
#             if success == True:
#                 ret,buffer=cv2.imencode('.jpg',frame)
#                 frame=buffer.tobytes()
#                 img = Image.open(io.BytesIO(frame))
#                 model = instantiate_model()
#                 results = model(img, size=640)
#                 results.print()
#                 img = np.squeeze(results.render())
#                 img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#             else:
#                 break
#             frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
#             FRAME_WINDOW.image(frame)
#     else:
#         cap.release()
#         cv2.destroyAllWindows()
#         st.warning('⚠ The Web-Camera is currently disabled. 😯')
#
# st.markdown("<br><hr><center>Made with ❤️ by <a href='mailto:ralhanprateek@gmail.com?subject=Automatic Number Plate Recognition WebApp!&body=Please specify the issue you are facing with the app.'><strong>Prateek Ralhan</strong></a></center><hr>", unsafe_allow_html=True)

#---------------------------------------------------------------------------

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import easyocr
from ultralytics import YOLO

UPLOAD_FOLDER = "static/"

# Load the YOLO model
model = YOLO('best.pt')

reader = easyocr.Reader(lang_list=['en'])

# Create a sidebar with three options
st.sidebar.title("Choose an option")
option = st.sidebar.radio("", ["Detect with Image", "Detect with Video", "Detect with IP Camera Live"])

# Main app content
st.title("Number Plate Detection")


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
    camera_ip = st.text_input("Enter the IP Camera URL")

    if camera_ip:
        st.write(f"Live feed from IP Camera: {camera_ip}")

        # Add code to display the live video feed from the IP camera
        # Perform number plate detection and text extraction in real-time
        pass

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