import pandas as pd
import numpy as np
import streamlit as st
import stdog
from PIL import Image
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input, decode_predictions

st.write("""
# Dog Classification Tool
The goal of this tool is to quickly predict a dog's breed based on a single image. \n
To run a prediction, upload a dog's image (try to ensure it's clear and only includes one dog) and click "Predict"
""")

######################
## Import the model ##
######################

@st.cache_resource
def load_model():

    dog_model = keras.saving.load_model("hf://abluna/dog_breed_v2")

    return dog_model

#########################
## Importing the image ##
#########################

img = st.file_uploader("Upload the image", type=None)

left_co,cent_co,last_co = st.columns(spec = [0.2,0.6,0.2])
with cent_co:
    if img is not None:
        original_image = Image.open(img)
        st.image(original_image, caption="Your Image", use_container_width=True)
 
###########################
## Importing Keras Model ##
###########################


dog_index_list = stdog.create_dog_index_list()
## dog_link_df = stdog.create_bird_image_links()

click_predict_message = "Predict Dog Breed"

if img is not None:
    if st.button(click_predict_message):
        with st.spinner("Wait for it..."):

            # Use the function to load your data
            tf_model = load_model()

            index_list = dog_index_list
            targ_size = 350

             # `img` is a PIL image of size 224x224
            img_v2 = image.load_img(img, target_size=(targ_size, targ_size))

            # `x` is a float32 Numpy array of shape (300, 300, 3)
            x = image.img_to_array(img_v2)

            # We add a dimension to transform our array into a "batch"
            # of size (1, 300, 300, 3)
            x = np.expand_dims(x, axis=0)

            # Finally we preprocess the batch
            # (this does channel-wise color normalization)
            x = preprocess_input(x)

            preds = tf_model.predict(x)

            ## Get list of predictions
            pred_dict = dict(zip(index_list, np.round(preds[0]*100,2)))
            Sorted_Prediction_Dictionary = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)

            Count_5Perc = preds[0][preds[0]>0.02]

            if len(Count_5Perc) == 1:
                TopPredictions = Sorted_Prediction_Dictionary[0]
                to_df = list(TopPredictions)
                df = pd.DataFrame({"Breed": to_df[0], "Probability":to_df[1]}, index=[0])
            if len(Count_5Perc) > 1:
                TopPredictions = Sorted_Prediction_Dictionary[0:len(Count_5Perc)]
                df = pd.DataFrame(TopPredictions, columns =["Breed", "Probability"])

            df["Probability"] = df["Probability"].round(2)
            #df = df.merge(bird_link_df, how="left", on="Breed")

            ## Get species and probability formatted

            df['Probability'] = df['Probability'].apply(lambda x: f"{x/100:.1%}")
            df['Caption'] = df['Breed'] + ' (' + df['Probability'] + ')'


            with cent_co:
                st.divider()
                st.markdown("##### :gray[Predicted Dog Breed (with % certainty):]")
                st.dataframe(df, hide_index=True)
                ##st.image(list(df["Link"]), caption = list(df["Caption"]), use_container_width=False)



            