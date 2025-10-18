import streamlit as st
import pandas as pd
from Multi_app_framework import MultiApp
st.set_page_config(layout='wide')
import Home, Dashboard, Make_Prediction # import your app modules here


#st.set_option('deprecation.showfileUploaderEncoding', False)
app = MultiApp()


# Add all your application here
app.add_app("Home", Home.app)
app.add_app("Dashboard", Dashboard.app)
app.add_app("Make a Prediction", Make_Prediction.app)
# The main app
app.run()







