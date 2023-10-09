#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
  
df = pd.read_csv('Iris_Froum14.csv') 
df.head() 


# In[4]:


# Dropping the Id column 
df.drop('Id', axis = 1, inplace = True) 
  
# Renaming the target column into numbers to aid training of the model 
df['Species']= df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}) 
  
# splitting the data into the columns which need to be trained(X) and the target column(y) 
X = df.iloc[:, :-1] 
y = df.iloc[:, -1] 
  
# splitting data into training and testing data with 30 % of data as testing data respectively 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
  
# importing the random forest classifier model and training it on the dataset 
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier() 
classifier.fit(X_train, y_train) 
  
# predicting on the test dataset 
y_pred = classifier.predict(X_test) 
  
# finding out the accuracy 
from sklearn.metrics import accuracy_score 
score = accuracy_score(y_test, y_pred)


# In[5]:


# pickling the model 
import pickle 
pickle_out = open("classifier.pkl", "wb") 
pickle.dump(classifier, pickle_out) 
pickle_out.close()


# In[6]:


import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
  
# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

  
def welcome(): 
    return 'welcome all'
  
# defining the function which will make the prediction using  
# the data which the user inputs    
    

def prediction(sepal_length, sepal_width, petal_length, petal_width):   
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]]) 
    print(prediction)
    return species_mapping[prediction[0]]  # Return the name of the species


  
# this is the main function in which we define our webpage  
def main(): 
      # giving the webpage a title 
    st.title("Iris Flower Prediction") 
      
    # here we define some of the front end elements of the web page like  
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Streamlit Iris Flower Classifier ML App </h1> 
    </div> 
    """
      
    # this line allows us to display the front end aspects we have  
    # defined in the above code 
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # the following lines create text boxes in which the user can enter  
    # the data required to make the prediction 
    sepal_length = st.text_input("Sepal Length", "Type Here") 
    sepal_width = st.text_input("Sepal Width", "Type Here") 
    petal_length = st.text_input("Petal Length", "Type Here") 
    petal_width = st.text_input("Petal Width", "Type Here") 
    result ="" 
    
    # the below line ensures that when the button called 'Predict' is clicked,  
    # the prediction function defined above is called to make the prediction  
    # and store it in the variable result 
    if st.button("Predict"): 
        result = prediction(float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)) 
    st.success('The predicted species of the iris flower is: {}'.format(result))

     
if __name__=='__main__': 
    main()   
