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
    lab1=Label(root,text='Beningnan',bg='powder blue',fg='black',font=('arial 16 bold')).pack()

    

ent1=Entry(root,text="HELLO",font=('arial 13')).pack()


but2=Button(root,text='Test Results',width=20,bg='brown',fg='white',command=predict).place(x=200,y=500)


root.mainloop()
