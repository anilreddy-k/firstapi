#Importing the libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

st.title('This model predicts if the given image is Aadhar POI or not')
#Loading the arguments
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# image_path = '/Users/anil/Downloads/sukshi/macro_model/sample_test_images/future/0aec124d780a632a7820f8ef52212d75_image.jpg'
model_path = 'macro_front_gray_rgb_19_07_v_0_1.h5'
height = 224
width = 224
interpolation = 'lanczos'

#Preprocessing the image (Converting the image to tensor)
def preprocessing_image(image_path):
    img = image.load_img(image_path, target_size = (height, width), interpolation = interpolation) #Load the image and resizes it to the specified size
    img_tensor = image.img_to_array(img) #Convert the image into numpy array
    img_tensor = np.expand_dims(img_tensor, axis=0) #(1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255. #The values are normalized between 0 and 1
    img.verify() #Verifying whether it is in fact an image
    return img_tensor

# image_name = image_path.split('/')[-1]
model = load_model(model_path) #Loading the model
# print(model.summary())

if uploaded_file:

    new_image = preprocessing_image(uploaded_file)
    pred_value = model.predict(new_image)

    st.write('predicted value is:', pred_value[0][0])

    if pred_value[0][0] < 0.5:
        st.write('this is aadhar POI')
    else:
        st.write('this is not aadhar POI')
