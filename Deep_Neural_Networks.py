import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) -0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:,1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)





n_pts=500
X,y=datasets.make_circles(n_samples=n_pts,random_state=123,noise=0.1,factor=0.2)
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])

model=Sequential()
model.add(Dense(4,input_shape=(2,),activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(Adam(lr=0.01),'binary_crossentropy',metrics=['accuracy'])
model.fit(x=X,y=y,verbose=1,batch_size=50,epochs=80,shuffle='true')






plot_decision_boundary(X, y, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])


X1=0.5
Y1=0.7

points=np.array([[X1,Y1]])
prediction=model.predict(points)
plt.plot([X1],[Y1],marker='o',color='b')
print("Prediction is :",prediction)
plt.show()
