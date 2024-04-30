import streamlit as st
from consume import predict_diabetes  # Import the function from apiendpoint.py

def main():
    st.set_page_config(page_title='Diabetes Prediction App', page_icon=':hospital:', layout='wide')

    st.title('Diabetes Prediction')
    st.markdown("""
    This is a simple app to predict diabetes based on provided features.
    Please enter patient details in the form below.
    """)

    # Create a sidebar for additional information or options
    st.sidebar.title('Options')
    st.sidebar.info("This is a demo version. The prediction functionality is disabled.")

    # Input fields for user to enter features
    st.header('Enter Patient Details')
    col1, col2 = st.columns(2)  # Use st.columns instead of st.beta_columns
    with col1:
        pregnancies = st.number_input('Pregnancies', value=0, step=1)
        plasma_glucose = st.number_input('Plasma Glucose', value=0)
        diastolic_blood_pressure = st.number_input('Diastolic Blood Pressure', value=0)
        triceps_thickness = st.number_input('Triceps Thickness', value=0)
    with col2:
        serum_insulin = st.number_input('Serum Insulin', value=0)
        bmi = st.number_input('BMI', value=0.0)
        diabetes_pedigree = st.number_input('Diabetes Pedigree', value=0.0)
        age = st.number_input('Age', value=0, step=1)

    # Prediction button
    if st.button('Predict', key='predict_button'):
        # Call the prediction function with the input values
        
        result = predict_diabetes(pregnancies, plasma_glucose, diastolic_blood_pressure, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age)
        
            
        if(result==1):
                st.success('You may have diabetes. Please consult a doctor!')
        elif(result==0):
                st.success('You may not have diabetes')
        else:
                st.error('Error in prediction please check the output logs')

if __name__ == '__main__':
    main()
