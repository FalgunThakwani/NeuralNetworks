import  numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import random


np.random.seed(0)


#60,000 train images and 10,000 test images
(X_train,y_train) , (X_test,y_test)=mnist.load_data()



#this will stop the code if the condition is not satisfied and throw an error
assert(X_train.shape[0]==y_train.shape[0]),"The number of images is not equal to the number of labels"
assert(X_test.shape[0]==y_test.shape[0]),"The number of images is not equal to the number of labels"
assert(X_train.shape[1:]==(28,28)),"The number of pixel is not equal to 28X28"
assert(X_test.shape[1:]==(28,28)),"The number of pixel is not equal to 28X28 "

number_of_sample=[]
cols=5
rows=10

fig , axs = plt.subplots(nrows=rows,ncols=cols,figsize=(5,10))
fig.tight_layout()

for i in range(cols):
    for j in range(rows):
        x_selected=X_train[y_train==j]


        #selects random image form 60,000 images

        axs[j][i].imshow(x_selected[random.randint(0,len(x_selected-1)),:,:],cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if(i==2):
            axs[j][i].set_title(str(j))





#hot_encoding labels
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_train,10)

#changing the intesity values of image to speed up the process
#255/255=1 0/255=0 ....

X_train=X_train/255
X_test=X_test/255


#changing the size to 1 row and 28X28 columns
number_of_pixels=784
X_train=X_train.reshape(X_train.shape[0],number_of_pixels)
X_test=X_test.reshape(X_test.shape[0],number_of_pixels)

plt.show()
