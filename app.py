import streamlit as st
import pickle
st.title('MPG ML PROJECT')
#displacement	horsepower	weight	acceleration
displacement = st.number_input(" displacement",value = 300,placeholder="enter value for displacement")
horsepower = st.number_input(" horsepower",value = 300,placeholder="enter value for horsepower")
acceleration = st.number_input(" acceleration",value = 300,placeholder="enter value for acceleration")
weight = st.number_input("weight",value = 300,placeholder="enter value for weight")
loaded_model = pickle.load(open('mpg_regression.sav','rb'))
prediction = loaded_model.predict([[displacement, horsepower,	weight	,acceleration]])
st.subheader(f'predicted mpg value for above parameter is {prediction}')
st.write(displacement,horsepower,weight,acceleration)
st.write(prediction)