import numpy as np
import activation as act

def output(W, An, O, Af, vi, L, Bias):
    i = 0
    for x in An:
        if(i == 0):
            An[i] = vi
        else:
            An[i] = np.matmul(An[i-1], W[i-1])
            
            j = 0
            
            An[i] = np.add(An[i], Bias[i])
            
            for y in x:
                if(i == L+1):
                    An[i][j] = act.actfunc(An[i][j], 2)
                else:
                    An[i][j] = act.actfunc(An[i][j], Af)
                j = j+1
        i = i+1

    return An

def erroratout(ve, va):
    return np.subtract(ve, va)

    