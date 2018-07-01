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

def mapFeature(X1, X2):
    degree=6
    out=np.matrix(np.ones(np.shape(X1)))
    for i in range(1,degree+1):
        for j in range(i+1):
            temp_col=np.multiply(np.power(X1,i-j),np.power(X2,j))
            out=np.hstack((out,temp_col))
    
    return out

def sigmoid(z):
    g=np.matrix(np.zeros(np.shape(z)))
    g=1/(1+np.exp(-1*z))
    return g


def costFunctionReg(theta,X,y,lamda):
    
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T
    m=np.shape(y)[0]
    J=0
    h=sigmoid(np.matmul(X,theta))
    temp_theta=np.multiply(theta,theta)
    temp_theta[0,0]=0
    J=np.sum((-1*np.multiply(y,np.log(h)))+(-1*np.multiply(1-y,np.log(1-h))))/m
    J=J+((lamda/(2*m))*np.sum(temp_theta))
    return J

def gradFunctionReg(theta,X,y,lamda):
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T   
    m=np.shape(y)[0]
    grad=np.matrix(np.zeros(np.shape(theta)))
    h=sigmoid(np.matmul(X,theta))
    temp_theta=np.matrix(np.zeros(np.shape(theta)))+theta
    temp_theta[0,0]=0
    grad=np.matmul((h-y).T,X).T/m
    grad=grad+((lamda/m)*temp_theta)
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


file_name="ex2data2.txt"
input_signal=np.loadtxt(file_name,delimiter=',')
X=np.matrix(input_signal[:,0:2])
y=np.matrix(input_signal[:,2]).T

plotData(X,y)

Xnew=mapFeature(X[:,0], X[:,1])
lamda = 0

(m,n)=np.shape(Xnew)
initial_theta=np.zeros(n)

cost=costFunctionReg(initial_theta,Xnew,y,lamda)
grad=gradFunctionReg(initial_theta,Xnew,y,lamda)

print("Cost at intial theta:")
print(cost)
print("Gradient at initial theta (zeros) for first five values:")
print(grad[0:5])

test_theta=np.ones(n)

cost=costFunctionReg(test_theta,Xnew,y,10)
grad=gradFunctionReg(test_theta,Xnew,y,10)

print("Cost at test theta:")
print(cost)
print("Gradient at initial theta (ones) for first five values:")
print(grad[0:5])



initial_theta=np.zeros(n)
lamda = 1

res = minimize(fun=costFunctionReg, x0=initial_theta, args=(Xnew,y,lamda), method='BFGS', jac=gradFunctionReg,options={'gtol': 1e-6, 'disp': True})
print("Theta trained:")
print(res.x)

p = predict(np.matrix(res.x).T, Xnew)
print("Train Accuracy:")
print(np.mean(p==y)*100)
