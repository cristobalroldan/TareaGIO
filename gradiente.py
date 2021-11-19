import numpy as np 
import math

m = 100
n = 2
a = np.random.uniform(-10,10, size=(m,n))
np.around(a,2,a)
b = np.random.uniform(-10,10, m)
np.around(b,2,b)
x = np.random.randint(1,2, n)


def f(x): 
    i = 1
    j = 1
    totalexp = 0
    totalsum = 0
    while i <= m:
        while j <= n:
            aux = a[i-1][j-1]*x[j-1] + b[i-1]
            totalexp = totalexp + aux
            j = j + 1
        aux2 = math.exp(totalexp)
        totalsum = totalsum + aux2
        # print(totalsum)
        i = i + 1
    return np.log(totalsum)

def fsinLog(x): 
    i = 1
    j = 1
    totalexp = 0
    totalsum = 0
    while i <= m:
        while j <= n:
            aux = a[i-1][j-1]*x[j-1] + b[i-1]
            totalexp = totalexp + aux
            j = j + 1
        aux2 = math.exp(totalexp)
        totalsum = totalsum + aux2
        # print(totalsum)
        i = i + 1
    return totalsum

def fdx(x):
    i = 1
    j =1
    totalexp = 0
    totalsum = 0
    while i <= m:
        while j <= n:
            aux = a[i-1][j-1]*x[j-1] + b[i-1]
            totalexp = totalexp + aux
            aux2 = math.exp(totalexp)*a[i-1][j-1]
            j = j +1

        totalsum = totalsum + aux2
        print(totalsum)
        i = i + 1
    
    return (1/fsinLog(x))*totalsum


def backtrack4(x0, f, fdx, t = 1, alpha = 0.2, beta = 0.8):
    
    point = np.asmatrix(x0) #Necessary to ensure matrix form
    while f(point - np.dot(t,fdx(point).T)) > f(point) + alpha * t * np.asscalar(np.dot(fdx(point).T, -1*fdx(point))):
         t *= beta
    return t

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
print(grad(x, 100))

# f(x)
# print(f(x))
# print(b)
# print(a[0][1) 
