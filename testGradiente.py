###################################################################
###################### EXAMPLE - 2 ################################
###################################################################
import numpy as np
import numpy.linalg as npla
#Randomly Initializing A and b. A is 100*3 matrix , b is a 100*1 matrix
A = np.asmatrix(np.random.rand(100,3))
b = np.asmatrix(np.random.rand(100,1))
#Defining the Function
def f(x): #x is 3d point e.g [1,1,1], Returns a scalar function value
    point = np.asmatrix(x)
    return np.asscalar(0.5*np.dot((np.dot(A,point.T) - b).T,(np.dot(A,point.T) - b)))
# first order derivatives of the function at point x
def fdx(x):
    point = np.asmatrix(x)
    return np.dot(np.dot(A.T,A),point.T) - np.dot(A.T,b) #Returns a matrix 3*1
# second order derivatives of the function at point x
def hess():
    
    return np.dot(A.T,A) #This is always Positive Definite, Returns a 3*3 matrix
###################################################################
###################### ANALYTICAL METHOD ##########################
###################################################################

#Analytical Solution - As it turns out linear algebra gives us an analytical solution. We will see how well numerical methods approximate the analytical solution
J = npla.pinv(np.dot(A.T,A))
print("Optimal x = ",np.dot(np.dot(J,A.T),b))
###################################################################
###################### LINE SEARCH STEP SIZE ######################
###################################################################
# This is an implementation of the backtracking algorithm that automatically generates a Step Size for every iteration at a point x0. This means you do not have to specify an arbitrary step size.
def backtrack4(x0, f, fdx, t = 1, alpha = 0.2, beta = 0.8):
    
    point = np.asmatrix(x0) #Necessary to ensure matrix form
    while f(point - np.dot(t,fdx(point).T)) > f(point) + alpha * t * np.asscalar(np.dot(fdx(point).T, -1*fdx(point))):
         t *= beta
    return t
###################################################################
###################### GRAD. DESCENT #############################
###################################################################
def grad(x0, max_iter):
    iter = 1
        
    while (np.linalg.norm(np.array(fdx(x0).flatten())[0]) > 0.000001):
    #Find stepsize by backtracking
        t = backtrack4(x0, f, fdx) #Step Size
        x0 = x0 - np.dot(t, fdx(x0).T)
        #Calculate New Value of Function
        print(x0, f(x0), fdx(x0), iter)
        iter += 1
        if iter > max_iter:
            break
    return x0, f(x0), iter

#Test
grad([0.5,0.5,0.5], 100)


#Your final optimal point will be very close to what you find through analytical solution