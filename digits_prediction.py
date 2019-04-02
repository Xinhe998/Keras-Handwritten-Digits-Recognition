"""# 準備mnist資料"""

import numpy as np
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt
np.random.seed(10)

from keras.datasets import mnist
(x_train_image, y_train_label), \
(x_test_image, y_test_label) = mnist.load_data()

print('train data=',len(x_train_image))
print ('x_train_image:',x_train_image.shape)
print ('y_train_label:',y_train_label.shape)

"""# 自己上傳的手寫Test Data"""

import os
from PIL import Image
import PIL.ImageOps
import matplotlib.image as mpimg
import random

def readData(txt_url,isShuffle):
    with open(txt_url,'r') as txtFile:
      content = txtFile.readlines()
      #是否需要打亂順序讀取圖片
      if isShuffle == True:
        random.shuffle(content)
        dataLen = len(content)
        images = np.empty((dataLen,28,28),dtype=np.uint8)
        label = np.empty((dataLen,),dtype='uint8')
      for i in range(dataLen):
        line = content[i]
        imgPath = line.split()[0]
        #打開照片，轉成灰階
        im = Image.open(line.split()[0]).convert('L')
        #反白
        im = PIL.ImageOps.invert(im)
        arr = np.asarray(im,dtype=np.uint8)
        images[i,:,:] = arr
        label[i] = line.split()[1]
      return (images,label)
  
(x_my_test_image, y_my_test_label) = readData('handwritten_dataset.txt',True)

print('test data=',len(x_my_test_image))
print ('x_my_test_image:',x_my_test_image.shape)
print ('y_my_test_label:',y_my_test_label.shape)

"""# 資料預處理"""

x_Train =x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
x_My_Test = x_my_test_image.reshape(10, 784).astype('float32')

x_Train_normalize = x_Train/ 255
x_Test_normalize = x_Test/ 255
x_My_Test_normalize = x_My_Test/ 255

y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)
y_My_TestOneHot = np_utils.to_categorical(y_my_test_label)

"""# 模型訓練"""

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=x_Train_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=30, batch_size=200, verbose=2)

def show_train_history(train_history, train, validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train History')
  plt.ylabel(train)
  plt.xlabel('Epoch')
  plt.legend(['train','validation'], loc='upper left')
  plt.show()

show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

"""# 驗證模型準確度"""

accuracy = model.evaluate(x_Test_normalize, y_TestOneHot)
print("accuracy = ", accuracy[1])

"""# 預測"""

prediction_x_Train = model.predict_classes(x_Train)
prediction_x_Test = model.predict_classes(x_Test)
prediction_x_My_Test = model.predict_classes(x_My_Test)

import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" +str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
    
print("\n\n================training資料===================\n")
plot_images_labels_prediction(x_train_image,y_train_label,prediction_x_Train,0,10)
    
print("\n\n================mnist的測試資料===================\n")
plot_images_labels_prediction(x_test_image,y_test_label,prediction_x_Test,0,10)

print("\n\n================自己手寫的測試資料===================\n")
plot_images_labels_prediction(x_my_test_image,y_my_test_label,prediction_x_My_Test,0,10)

"""# 混淆矩陣"""

import pandas as pd
pd.crosstab(y_my_test_label, prediction_x_My_Test, rownames=['label'], colnames=['prediction'])

df = pd.DataFrame({'label': y_my_test_label, 'prediction': prediction_x_My_Test})

#找出7都會被誤判成6
df[(df.label==7)&(df.prediction==6)]

"""# 自己上傳資料的準確率"""

my_score = model.evaluate(x_My_Test_normalize,y_My_TestOneHot)
print("準確率:", my_score[1])