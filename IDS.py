
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd
import os
from sklearn import *
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Customized Neaural Network Based Intrusion Detection System")
main.geometry("1300x1200")

global filename
global labels 
global columns
global balance_data
global data
global X, Y, X_train, X_test, y_train, y_test
global svm_acc, random_acc, dnn_acc, eml_acc
global normal_time
global parallel_time

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def importdata(): 
    global balance_data
    balance_data = pd.read_csv("clean.txt") 
    return balance_data 

def splitdataset(balance_data): 
    X = balance_data.values[:, 0:37] 
    Y = balance_data.values[:, 38] 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test 

def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def preprocess(): 
    global labels
    global columns
    global filename
    
    text.delete('1.0', END)
    columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
               "num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
               "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
               "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
               "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
               "dst_host_srv_rerror_rate","label"]

    labels = {"normal":0,"neptune":1,"warezclient":2,"ipsweep":3,"portsweep":4,"teardrop":5,"nmap":6,"satan":7,"smurf":8,"pod":9,"back":10,"guess_passwd":11,
              "ftp_write":12,"multihop":13,"rootkit":14,"buffer_overflow":15,"imap":16,"warezmaster":17,"phf":18,"land":19,"loadmodule":20,"spy":21,"perl":22,
              "saint":23,"mscan":24,"apache2":25,"snmpgetattack":26,"processtable":27,"httptunnel":28,"ps":29,"snmpguess":30,"mailbomb":31,"named":32,"sendmail":33,
              "xterm":34,"worm":35,"xlock":36,"xsnoop":37,"sqlattack":38,"udpstorm":39}
    
    balance_data = pd.read_csv(filename)
    dataset = ''
    index = 0
    cols = ''
    for index, row in balance_data.iterrows():
      for i in range(0,42):
        if(isfloat(row[i])):
          dataset+=str(row[i])+','
          if index == 0:
            cols+=columns[i]+','
      dataset+=str(labels.get(row[41]))
      if index == 0:
        cols+='Label'
      dataset+='\n'
      index = 1;
    
    f = open("clean.txt", "w")
    f.write(cols+"\n"+dataset)
    f.close()
    
    text.insert(END,"Removed non numeric characters from dataset and saved inside clean.txt file\n\n")
    text.insert(END,"Dataset Information\n\n")
    text.insert(END,dataset+"\n\n")

    dr = pd.read_csv(filename)
    attacks = dr.values[:,dr.shape[1]-1]
    unique = np.unique(attacks)
    graph_col = []
    graph_row = []
    for i in range(len(unique)):
        count = np.count_nonzero(attacks == unique[i])
        graph_col.append(unique[i])
        graph_row.append(count)

    fig, ax = plt.subplots()
    y_pos = np.arange(len(graph_col))
    plt.bar(y_pos, graph_row)
    plt.xticks(y_pos, graph_col)
    ax.xaxis_date()
    fig.autofmt_xdate() 
    plt.show()        

def generateModel():
    global data
    global X, Y, X_train, X_test, y_train, y_test

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    text.delete('1.0', END)
    text.insert(END,"Training model generated\n\n")
    text.insert(END,"Total records found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"80% records used to train deep learning algorithm : "+str(len(X_train))+"\n")
    text.insert(END,"20% records used to train deep learning algorithm : "+str(len(X_test))+"\n")

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n")  
    fig, ax = plt.subplots()
    sns.heatmap(cm,annot=True,fmt='g')
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    fig.autofmt_xdate() 
    plt.show()  
    return accuracy 

def runSVM():
    global svm_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = svm.SVC() 
    cls.fit(X_train, y_train) 
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    svm_acc = cal_accuracy(y_test, prediction_data,'SVM Accuracy, Classification Report & Confusion Matrix') 

def runRandomForest():
    global random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=3,max_depth=0.9,random_state=None) 
    cls.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy, Classification Report & Confusion Matrix') 

def runDNN():
    global dnn_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Y = Y.astype('uint8')
    Y1 = to_categorical(Y)
    model = Sequential() #creating DNN model object
    model.add(Dense(256, input_dim=X.shape[1], activation='relu', kernel_initializer = "uniform")) #defining one layer with 256 filters to filter dataset
    model.add(Dense(128, activation='relu', kernel_initializer = "uniform"))#defining another layer to filter dataset with 128 layers
    model.add(Dense(Y1.shape[1], activation='softmax',kernel_initializer = "uniform")) #after building model need to predict attack
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #while filtering and training dataset need to display accuracy 
    print(model.summary()) #display rnn details
    dnn_acc = model.fit(X, Y1, epochs=10, batch_size=64) #start building DNN model
    values = dnn_acc.history #save each epoch accuracy and loss
    values = values['accuracy']
    dnn_acc = values[9] * 100
    #text.insert(END,"DNN Accuracy : "+str(dnn_acc)+"\n\n")
    prediction_data=model.predict(X_test)
    prediction_data = np.argmax(prediction_data,axis=1)
    DNN_acc = cal_accuracy(y_test, prediction_data,'DNN Algorithm Accuracy, Classification Report & Confusion Matrix') 
    

def graph():
    height = [svm_acc,random_acc,dnn_acc]
    bars = ('SVM', 'Random Forest','DNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()



font = ('times', 16, 'bold')
title = Label(main, text='INNOVATIVE INTRUSION DETECTION FOR IOT: A RESOURCE EFFICIENT NEURAL NETWORK APPROACH')
title.config(bg='LightSteelBlue1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload NSL-KDD Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='black')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

preprocess = Button(main, text="Preprocessing", command=preprocess)
preprocess.place(x=50,y=150)
preprocess.config(font=font1) 

model = Button(main, text="Generate Training Model", command=generateModel)
model.place(x=200,y=150)
model.config(font=font1) 

runsvm = Button(main, text="Apply SVM", command=runSVM)
runsvm.place(x=475,y=150)
runsvm.config(font=font1) 

runrandomforest = Button(main, text="Apply Random Forest", command=runRandomForest)
runrandomforest.place(x=620,y=150)
runrandomforest.config(font=font1) 

rundnn = Button(main, text="Apply DNN Model", command=runDNN)
rundnn.place(x=50,y=200)
rundnn.config(font=font1)

graph = Button(main, text="Accuracy Graph", command=graph)
graph.place(x=550,y=200)
graph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='light cyan')
main.mainloop()
