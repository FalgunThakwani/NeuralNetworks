import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


n_pts=500
np.random.seed(0)
top_region=np.array([np.random.normal(13,2,n_pts),np.random.normal(12,2,n_pts)]).T
bottom_region=np.array([np.random.normal(5,2,n_pts),np.random.normal(6,2,n_pts)]).T
X=np.vstack((top_region,bottom_region))
Y=np.matrix(np.append(np.zeros(n_pts),np.ones(n_pts))).T

plt.scatter(top_region[:,0],top_region[:,1])
plt.scatter(bottom_region[:,0],bottom_region[:,1])


model=Sequential()
model.add(Dense(units=1 , input_shape=(2,) , activation='sigmoid'))
adam=Adam(lr=0.1)
model.compile(adam,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x=X,y=Y,verbose=0,batch_size=50,epochs=100,shuffle='true')



plot_decision_boundary(X, Y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

x = 7.5
y = 5


point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("prediction is: ",prediction)

plt.show()
