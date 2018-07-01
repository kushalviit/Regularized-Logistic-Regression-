import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import minimize, rosen, rosen_der

def plotData(X,y):
    pos_index=np.where(y==1)[0]
    neg_index=np.where(y==0)[0]
    pos_X=X[pos_index,:]
    neg_X=X[neg_index,:]
    plt.figure()
    plt.scatter(pos_X[:,0],pos_X[:,1],marker='^')
    plt.scatter(neg_X[:,0],neg_X[:,1],marker='o')
    plt.show()

def sigmoid(z):
    g=np.matrix(np.zeros(np.shape(z)))
    g=1/(1+np.exp(-1*z))
    return g
    
def costFunction(theta,X,y):
    
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T
    m=np.shape(y)[0]
    J=0
    h=sigmoid(np.matmul(X,theta))
    J=np.sum((-1*np.multiply(y,np.log(h)))+(-1*np.multiply(1-y,np.log(1-h))))/m
    return J

def gradFunction(theta,X,y):
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T   
    m=np.shape(y)[0]
    grad=np.matrix(np.zeros(np.shape(theta)))
    h=sigmoid(np.matmul(X,theta))
    grad=np.matmul((h-y).T,X).T/m
    return np.squeeze(np.asarray(grad))

def predict(theta,X):
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T
    X=np.matrix(X)
    (m,n)=np.shape(X)
    p =np.matrix(np.zeros((m, 1)))
    t_p=sigmoid(np.matmul(X,theta))
    pos_ind=np.where(t_p>=0.5)
    p[pos_ind,:]=1
    return p  

file_name="ex2data1.txt"
input_signal=np.loadtxt(file_name,delimiter=',')
X=np.matrix(input_signal[:,0:2])
y=np.matrix(input_signal[:,2]).T

plotData(X,y)

(m,n)=np.shape(X)


X=np.hstack((np.transpose(np.matrix(np.ones(m))),X))
initial_theta=np.zeros(n+1)


cost = costFunction(initial_theta, X, y)
grad = gradFunction(initial_theta, X, y)


print("Cost at intial theta:")
print(cost)
print("Gradient at initial theta (zeros):")
print(grad)

test_theta = np.matrix(np.array([-24, 0.2, 0.2])).T
cost = costFunction(test_theta, X, y)
grad = gradFunction(test_theta, X, y)
print("Cost at intial theta:")
print(cost)
print("Gradient at initial theta (zeros):")
print(grad)


res = minimize(fun=costFunction, x0=initial_theta, args=(X,y), method='BFGS', jac=gradFunction,options={'gtol': 1e-6, 'disp': True})
print("Theta trained:")
print(res.x)


prob = sigmoid(np.matmul(np.matrix([1,45,85]) , np.matrix(res.x).T));
print("For a student with scores 45 and 85, we predict an admission probability of ", prob);
print("Expected value: 0.775 +/- 0.002");

#Compute accuracy on our training set
p = predict(np.matrix(res.x).T, X)
print("Train Accuracy:")
print(np.mean(p==y)*100)

