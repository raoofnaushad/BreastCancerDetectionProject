import tkinter as tk
from tkinter import *
# from PIL import Image

import pandas as pd
import numpy as np
import  seaborn as sns
import matplotlib.pyplot as plt


from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


window=tk.Tk()
window.geometry("800x800")
window.title(" Breast Cancer Detector- Final Year Project ")
# window.config(background='powder blue')


radius = tk.Label(text = "Mean Radius")
radius.grid(column=0,row=1)
texture = tk.Label(text = "Mean Texture")
texture.grid(column=0,row=2)
perimeter = tk.Label(text = "Mean Perimeter")
perimeter.grid(column=0,row=3)
area = tk.Label(text = "Mean Area")
area.grid(column=0,row=4)
#
smoothness = tk.Label(text = "Mean Smoothness")
smoothness.grid(column=0,row=5)
compactness = tk.Label(text = "Mean compactness")
compactness.grid(column=0,row=6)
concavity = tk.Label(text = "Mean concavity")
concavity.grid(column=0,row=7)
concave = tk.Label(text = "Mean concave points")
concave.grid(column=0,row=8)

Symmetry = tk.Label(text = "Mean Symmetry")
Symmetry.grid(column=0,row=9)
Fractal = tk.Label(text = "Mean Fractal Dimension")
Fractal.grid(column=0,row=10)
Radius1 = tk.Label(text = "STD err Radius")
Radius1.grid(column=0,row=11)
Texture1 = tk.Label(text = "STD err Texture")
Texture1.grid(column=0,row=12)


######################
perimeter1 = tk.Label(text = "STD err Perimeter")
perimeter1.grid(column=0,row=13)
area1 = tk.Label(text = "STD err Area")
area1.grid(column=0,row=14)
#
smoothness1 = tk.Label(text = "STD err Smoothness")
smoothness1.grid(column=0,row=15)
compactness1 = tk.Label(text = "STD err compactness")
compactness1.grid(column=2,row=1)
concavity1 = tk.Label(text = "STD err concavity")
concavity1.grid(column=2,row=2)
concave1 = tk.Label(text = "STD err concave points")
concave1.grid(column=2,row=3)

Symmetry1 = tk.Label(text = "STD err Symmetry")
Symmetry1.grid(column=2,row=4)
Fractal1 = tk.Label(text = "STD err Fractal Dimension")
Fractal1.grid(column=2,row=5)
Radius2 = tk.Label(text = "worst Radius")
Radius2.grid(column=2,row=6)
Texture2 = tk.Label(text = "worst Texture")
Texture2.grid(column=2,row=7)

###############################
perimeter2 = tk.Label(text = "worst Perimeter")
perimeter2.grid(column=2,row=8)
area2 = tk.Label(text = "worst Area")
area2.grid(column=2,row=9)
#
smoothness2 = tk.Label(text = "worst Smoothness")
smoothness2.grid(column=2,row=10)
compactness2 = tk.Label(text = "worst compactness")
compactness2.grid(column=2,row=11)
concavity2 = tk.Label(text = "worst concavity")
concavity2.grid(column=2,row=12)
concave2 = tk.Label(text = "worst concave points")
concave2.grid(column=2,row=13)

Symmetry2 = tk.Label(text = "worst Symmetry")
Symmetry2.grid(column=2,row=14)
Fractal2 = tk.Label(text = "worst Fractal Dimension")
Fractal2.grid(column=2,row=15)


#
radiusEntry = tk.Entry()
radiusEntry.grid(column=1,row=1)
textureEntry = tk.Entry()
textureEntry.grid(column=1,row=2)
perimeterEntry = tk.Entry()
perimeterEntry.grid(column=1,row=3)
areaEntry = tk.Entry()
areaEntry.grid(column=1,row=4)

smoothnessEntry = tk.Entry()
smoothnessEntry.grid(column=1,row=5)
compactnessEntry = tk.Entry()
compactnessEntry.grid(column=1,row=6)
concavityEntry = tk.Entry()
concavityEntry.grid(column=1,row=7)
concaveEntry = tk.Entry()
concaveEntry.grid(column=1,row=8)

symmetryEntry = tk.Entry()
symmetryEntry.grid(column=1,row=9)
fractalEntry = tk.Entry()
fractalEntry.grid(column=1,row=10)
radiusEntry1 = tk.Entry()
radiusEntry1.grid(column=1,row=11)
textureEntry1 = tk.Entry()
textureEntry1.grid(column=1,row=12)

###################################
perimeterEntry1 = tk.Entry()
perimeterEntry1.grid(column=1,row=13)
areaEntry1 = tk.Entry()
areaEntry1.grid(column=1,row=14)

smoothnessEntry1 = tk.Entry()
smoothnessEntry1.grid(column=1,row=15)
compactnessEntry1 = tk.Entry()
compactnessEntry1.grid(column=4,row=1)
concavityEntry1 = tk.Entry()
concavityEntry1.grid(column=4,row=2)
concaveEntry1 = tk.Entry()
concaveEntry1.grid(column=4,row=3)

symmetryEntry1 = tk.Entry()
symmetryEntry1.grid(column=4,row=4)
fractalEntry1 = tk.Entry()
fractalEntry1.grid(column=4,row=5)
radiusEntry2 = tk.Entry()
radiusEntry2.grid(column=4,row=6)
textureEntry2 = tk.Entry()
textureEntry2.grid(column=4,row=7)

################################################\\
###################################
perimeterEntry2 = tk.Entry()
perimeterEntry2.grid(column=4,row=8)
areaEntry2 = tk.Entry()
areaEntry2.grid(column=4,row=9)

smoothnessEntry2 = tk.Entry()
smoothnessEntry2.grid(column=4,row=10)
compactnessEntry2 = tk.Entry()
compactnessEntry2.grid(column=4,row=11)
concavityEntry2 = tk.Entry()
concavityEntry2.grid(column=4,row=12)
concaveEntry2 = tk.Entry()
concaveEntry2.grid(column=4,row=13)

symmetryEntry2 = tk.Entry()
symmetryEntry2.grid(column=4,row=14)
fractalEntry2 = tk.Entry()
fractalEntry2.grid(column=4,row=15)




def getInput():
    list = []
    list.append(float(radiusEntry.get()))
    list.append(float(textureEntry.get()))
    list.append(float(perimeterEntry.get()))
    list.append(float(areaEntry.get()))
    list.append(float(smoothnessEntry.get()))
    list.append(float(compactnessEntry.get()))
    list.append(float(concavityEntry.get()))
    list.append(float(concaveEntry.get()))
    list.append(float(symmetryEntry.get()))
    list.append(float(fractalEntry.get()))


    list.append(float(radiusEntry1.get()))
    list.append(float(textureEntry1.get()))
    list.append(float(perimeterEntry1.get()))
    list.append(float(areaEntry1.get()))
    list.append(float(smoothnessEntry1.get()))
    list.append(float(compactnessEntry1.get()))
    list.append(float(concavityEntry1.get()))
    list.append(float(concaveEntry1.get()))
    list.append(float(symmetryEntry1.get()))
    list.append(float(fractalEntry1.get()))
    
    list.append(float(radiusEntry2.get()))
    list.append(float(textureEntry2.get()))
    list.append(float(perimeterEntry2.get()))
    list.append(float(areaEntry2.get()))
    list.append(float(smoothnessEntry2.get()))
    list.append(float(compactnessEntry2.get()))
    list.append(float(concavityEntry2.get()))
    list.append(float(concaveEntry2.get()))
    list.append(float(symmetryEntry2.get()))
    list.append(float(fractalEntry2.get()))
    
    list1 = []
    list1.append(list)
    
    # print(list1)
    
    model = load_model('my_model.h5')
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(data = cancer.data, columns=cancer.feature_names)
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
    X_test = X_test.append(pd.DataFrame(list1, columns=cancer.feature_names),ignore_index=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_test1 = scaler.transform(X_test1)
    # print(X_test)
    # print(type(X_test))
    # toPredict = scaler.transform(toPredict)
    
    X_train = X_train.reshape(455,30,1)
    X_test = X_test.reshape(115, 30, 1)
    # X_test1 = X_test1.reshape(1, 30, 1)
    # X_test = X_test.reshape(114, 30, 1)

    print(np.exp(model.predict(X_test[114:])))
    predicted = np.exp(model.predict(X_test[114:]))
    print(predicted)
    for i in range(0,1):
        if predicted[0][0] >= 2:
            print("Cancer is detected")
            answer = "Cancer is detected"
        else:
            print("Cancer is not detected")
            answer = "Cancer is not detected"
    
    textArea = tk.Text(master=window,height=5,width=25)
    textArea.grid(column=1,row=40)
    # answer = " Heyy {monkey}!!!. You are {age} years old!!! ".format(monkey=name, age=monkey.age())
    textArea.insert(tk.END,answer)
    
    
    # name=nameEntry.get()
    # monkey = Person(name,datetime.date(int(yearEntry.get()),int(monthEntry.get()),int(dateEntry.get())))
    # textArea = tk.Text(master=window,height=10,width=25)
    # textArea.grid(column=1,row=6)
    # answer = " Heyy {monkey}!!!. You are {age} years old!!! ".format(monkey=name, age=monkey.age())
    # textArea.insert(tk.END,answer)
    pass

button=tk.Button(window,text="Test result",command=getInput,bg="pink")
button.grid(column=1,row=35)



lab1=Label(window,text="Breast Cancer Detector", font=('arial 16 bold'))
lab1.grid(column=1, row=0)


window.mainloop()