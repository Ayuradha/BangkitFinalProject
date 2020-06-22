import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import  load_img, img_to_array
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import clear_session
import pandas as pd



st.sidebar.title("About")



st.sidebar.info(

    "This is a demo application written to help you understand Streamlit. The application identifies the skin leison in the picture. It was built using a Convolution Neural Network (CNN).")

@st.cache
def data_gen(x):
    img = np.asarray(Image.open(x).resize((100, 75)))
    x_test = np.asarray(img.tolist())
    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)
    x_test = (x_test - x_test_mean) / x_test_std
    x_validate = x_test.reshape(1, 75, 100, 3)

    return x_validate

def load_models():
    model_weights = 'modelb.h5'
    model_json = 'modelb.json'
    with open(model_json) as json_file:
        lmodel = model_from_json(json_file.read())
        lmodel.load_weights(model_weights)
    return lmodel


@st.cache(allow_output_mutation=True)
def get_database_connection():
    return db.get_connection()
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = model.predict_proba(x_test)
    clear_session()
    ynew = np.round(ynew, 2)
    ynew = ynew
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    clear_session()
    return y_new, Y_pred_classes


@st.cache
def display_prediction(y_new):
    """Display image and preditions from model"""

    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {2: 'Benign keratosis-like lesions', 4: 'Melanocytic nevi', 3: 'Dermatofibroma',
                        5: 'Melanoma', 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result

def main():
    uploaded_file = st.file_uploader("Choose an image...", type=("jpg","png","jpeg"))
    st.text("Please upload skin | only jpg & jpeg & png files")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        if st.checkbox('Predict'):
            model=load_models()
            x_test = data_gen(uploaded_file)
            y_new, Y_pred_classes = predict(x_test, model)
            result = display_prediction(y_new)
            st.write(result)
          
if __name__ == "__main__":
    main()
