#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

data = pd.read_csv('diabetes.csv')

X = data.drop(['Outcome'],axis=1)
y = data['Outcome']
scaler = StandardScaler()
standard_data = scaler.fit_transform(X)

X = standard_data
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = svm.SVC(kernel='linear')
model.fit(X_train,y_train)

predict_X_train = model.predict(X_train)
X_train_accuracy = accuracy_score(predict_X_train, y_train)

predict_X_test = model.predict(X_test)
X_test_accuracy = accuracy_score(predict_X_test, y_test)


input_data = (0,137,40,35,168,43.1,2.288,33)

input_data_to_array = np.asarray(input_data)

input_data_reshape = input_data_to_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

print(std_data)

prediction = model.predict(std_data)
print(prediction)

st.title('Diabetes Prediction')

st.header('Input Data')
pregnancies = st.number_input('Number of Pregnancies', 0, 17, 0)
glucose = st.number_input('Glucose Level')
blood_pressure = st.number_input('Blood Pressure (mm Hg)', 0, 122, 72)
skin_thickness = st.number_input('Skin Thickness (mm)', 0, 99, 20)
insulin = st.number_input('Insulin (mu U/ml)', 0, 846, 79)
bmi = st.number_input('BMI', 0.0, 67.1, 32.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
age = st.number_input('Age (years)', 21, 81, 29)

input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])

std_input_data = scaler.transform(input_data.reshape(1, -1))

if st.button('Predict'):
    prediction = model.predict(std_input_data)
    if prediction[0] == 1:
        st.error('You may have diabetes.')
    else:
        st.success('You may not have diabetes.')


st.sidebar.markdown('Model Information:')
st.sidebar.write("This model is based on a Support Vector Machine (SVM) classifier with a linear kernel.")
st.sidebar.markdown('Predictive Model')


