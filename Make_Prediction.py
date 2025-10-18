import streamlit as st
import numpy as np
import onnxruntime as ort

# --- Load ONNX model once and reuse ---
@st.cache_resource
def get_onnx_session(model_path="stroke_model.onnx"):
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

sess = get_onnx_session()

def stroke_prediction(input_data):
    X = np.array([input_data], dtype=np.float32)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X})
    label = outputs[0]
    pred = int(np.ravel(label)[0])
    if pred == 0:
        return "This person is not likely to have a stroke."
    else:
        return "This person is likely to have a stroke."


def app():
    st.title(" Predicting Stroke likelihood :orange[System] :bar_chart:")
    st.write("You are in 'predict' page.")
    st.markdown(
        "This Analysis dashboard lets us explore the relationship between **stroke** and the various "
        "variables in the dataset."
    )

    # Getting user input
    sex = st.selectbox('Select Gender', ['Male', 'Female', 'Other'])
    age = st.slider('Age of the person', min_value=0, max_value=120, value=30)
    high_bp_sel = st.selectbox('High blood pressure?', ['Yes', 'No'])
    heart_disease_sel = st.selectbox('Heart disease?', ['Yes', 'No'])
    ever_married = st.selectbox('Ever been married?', ['Yes', 'No'])
    job_type = st.selectbox('Job type', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'])
    settlement = st.selectbox('Settlement type', ['Urban', 'Rural'])
    avg_sugar_level = st.slider('Average sugar/glucose level', min_value=0, max_value=500, value=100)
    body_mass_indx = st.slider('Body Mass Index (BMI)', min_value=0, max_value=100, value=25)
    smoking_status = st.selectbox('Smoking status', ['Smokes', 'Never Smoked', 'Unknown'])

    # Encode inputs
    gender_dict = {'Male': 1, 'Female': 2, 'Other': 3}
    married_dict = {'Yes': 1, 'No': 0}
    work_dict = {'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'Children': 4, 'Never_worked': 5}
    residence_dict = {'Urban': 1, 'Rural': 0}
    smoking_dict = {'Smokes': 1, 'Never Smoked': 2, 'Unknown': 0}
    high_bp_dict = {'Yes': 1, 'No': 0}
    heart_disease_dict = {'Yes': 1, 'No': 0}

    gender = gender_dict[sex]
    married = married_dict[ever_married]
    work = work_dict[job_type]
    residence = residence_dict[settlement]
    smoking = smoking_dict[smoking_status]
    high_bp = high_bp_dict[high_bp_sel]
    heart_disease = heart_disease_dict[heart_disease_sel]

    diagnosis = ''
    if st.button('Stroke likelihood'):
        input_data = (
            gender, age, high_bp, heart_disease, married, work, residence,
            avg_sugar_level, body_mass_indx, smoking
        )
        diagnosis = stroke_prediction(input_data)

    if diagnosis:
        st.success(diagnosis)


if __name__ == "__main__":
    app()
