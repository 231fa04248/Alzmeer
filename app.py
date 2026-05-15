import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
from utils import *

# Initialize DB
create_table()

# Load Model
model = load_model("model/alzheimer_model.h5")

classes = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# Session state
if "login_status" not in st.session_state:
    st.session_state.login_status = False

# ---------------- UI ---------------- #
st.set_page_config(page_title="Alzheimer System", layout="wide")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------- REGISTER ---------------- #
if choice == "Register":
    st.title("Create Account")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type='password')

    if st.button("Register"):
        add_user(new_user, new_pass)
        st.success("Account created successfully!")

# ---------------- LOGIN ---------------- #
elif choice == "Login":
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        result = login_user(username, password)

        if result:
            st.session_state.login_status = True
            st.success(f"Welcome {username}")
        else:
            st.error("Invalid credentials")

# ---------------- MAIN APP ---------------- #
if st.session_state.login_status:

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", " Prediction", "Dashboard", "Logout"])

    # -------- HOME -------- #
    if page == " Home":
        st.title(" Alzheimer Detection System")
        st.write("Upload MRI images to detect Alzheimer's stage using AI.")

    # -------- PREDICTION -------- #
    elif page == " Prediction":
        st.title(" Upload MRI Image")

        uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", use_column_width=True)

            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)
            predicted_class = classes[np.argmax(pred)]
            confidence = float(np.max(pred))

            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence*100:.2f}%")

            # Probability chart
            df = pd.DataFrame(pred, columns=classes)
            st.bar_chart(df.T)

    # -------- DASHBOARD -------- #
    elif page == "Dashboard":
        st.title("Model Performance")

        st.image("assets/accuracy.png", caption="Accuracy")
        st.image("assets/loss.png", caption="Loss")
        st.image("assets/confusion_matrix.png", caption="Confusion Matrix")

    # -------- LOGOUT -------- #
    elif page == "Logout":
        st.session_state.login_status = False
        st.success("Logged out successfully")