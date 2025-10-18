import streamlit as st
from PIL import Image
def app():
    st.title(" Predicting Stroke likelihood :orange[Dashboard] :bar_chart:")
    st.markdown("You are at the 'home' page")
    st.markdown("This Analysis dashboard lets us explore the relationship between **stroke** and the various variables in the dataset")

    st.markdown("According to National Heart, Lung and Blood Institute, a stroke occurs when brain cells die suddenly due to insufficient blood flow and oxygen caused by either a blood clot blocking an artery in the brain or a blood vessel rupturing. ")
    st.markdown("Early detection of a stroke can often save the individual's life. ")
    st.markdown("Carrying out descriptive statistics will allow the researcher to explore some of the machine learning and data science libraries in Python to extract meaningful insights from large a given amounts of data.")

    # Load the image
    img = Image.open('pngwing.com-2.png')

    # Display the image
    st.image(img, caption='My Image')
