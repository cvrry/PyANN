import math
import numpy as np

########################################

def tanh(x):
  return np.tanh(x)

def logistic(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

########################################

def itanh(x):
  return np.arctanh(x)

def ilogistic(x):
  return np.log(x/(1-x+0.0001))

def irelu(x):
  return x

########################################

def dtanh(x):
  return 1 - math.pow(tanh(x), 2)

def dlogistic(x):
  y = logistic(x)
  return (1 - y)*y

def drelu(x):
  if(x > 0):
    return 1
  else:
    return 0

################################################################################

def actfunc(val, func):

    if(func == 1):
      return tanh(val)
    
    elif(func == 2):
        return logistic(val)

    elif(func == 3):
        return relu(val)

def iactfunc(val, func):

    if(func == 1):
      return itanh(val)
    
    elif(func == 2):
        return ilogistic(val)

    elif(func == 3):
        return irelu(val)

def dactfunc(val, func):

    if(func == 1):
      return dtanh(val)
    
    elif(func == 2):
        return dlogistic(val)

    elif(func == 3):
        return drelu(val)
