import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
# from keras.models import load_model

from tensorflow.keras.optimizers import Adam


import pandas as pd
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt


from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tkinter import *
root=Tk()
root.geometry('800x800')
root.title("SCT Engineering College")
lab1=Label(root,text='Breast Cancer Detection Using CNN',bg='powder blue',fg='black',font=('arial 16 bold')).pack()
root.config(background='powder blue')

lab2=Label(root,text='Enter details',font=('arial 16'),bg='powder blue',fg='black').pack()


def predict():
    
    model = load_model('my_model.h5')
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train.reshape(455,30,1)
    X_test = X_test.reshape(114, 30, 1)
    print(np.exp(model.predict(X_test[0:8])))
    ls = input1.get()
    lab1=Label(root,text=ls,bg='powder blue',fg='black',font=('arial 16 bold')).pack()
    

    


inLa=Label(root,text="mean radius",bg='powder blue',fg='black').pack()
input1 = Entry(root,width=10)
input1.pack()
input2 = Entry(root,width=10)
input2.pack()
input3 = Entry(root,width=10)
input3.pack()

but2=Button(root,text='Test Results',width=20,bg='brown',fg='white',command=predict).place(x=200,y=500)


root.mainloop()
