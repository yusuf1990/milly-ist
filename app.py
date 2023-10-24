import numpy as np
import streamlit as st
import pickle
import lightgbm

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data=pickle.load(file)
    return data

data=load_model()
classifier=data['model']
vectorizer=data['vectorizer']

def show_prediction():
    st.title("Knownn Fraud Detection Application")

    st.write("### Enter the text  to determine if it's spam or not")
    message=st.text_area("Enter the Email/SMS massage:","")

    ok=st.button("Check if it's spam or not")
    if ok:
        X=np.array([message])
        X_str=np.vectorize(str)(X)
        transformed_data = data['vectorizer'].transform(X_str)

        prediction=classifier.predict_proba(transformed_data)[:, 1]
        #st.subheader(f"The client will renew its claim{detector}")
        #st.subheader(f"{prediction}")
        # Display the result
        if prediction >= 0.3: # 'spam':
            st.subheader("This message is spam.")
        else:
            st.subheader("This message is not spam.")
    elif ok and not message:
        st.warning("Please enter a message before checking.")

show_prediction()