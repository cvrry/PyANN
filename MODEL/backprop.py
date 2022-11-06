import activation as act
import numpy as np

def backprop(dW, dB, W, Bias, L, An, Err, lrate, mom, Af, I, N, O, reg):
    
    #calculating_Local_Gradients
    LG = []
    LG.append( np.zeros(I))

    for i in range(1, L+1):
        LG.append( np.zeros(N))

    LG.append( np.zeros(O))

    i = L+1
    for x in LG:
        j = 0
        for y in LG[i]:
            if(i == L+1):
                LG[i][j] = (Err[j]*act.dactfunc(act.iactfunc(An[i][j], 2), 2))

            else:
                k = 0
                e = 0
                for z in LG[i+1]:
                    e = e + (LG[i+1][k]*W[i][j][k])
                    k = k+1

                LG[i][j] = (e*act.dactfunc(act.iactfunc(An[i][j], Af), Af))
            j = j + 1
        i = i - 1

    for i in range(L, 0, -1):
        j = 0
        for x in dW[i]:
            M = np.size(dW[i])
            k = 0
            for y in x:
                dW[i][j][k] = mom*dW[i][j][k] + (1-mom)*lrate*An[i][j]*LG[i+1][k] - (reg*dW[i][j][k]/M)
                k = k + 1
            
            M = np.size(dB[i])
            dB[i][j] = mom*dB[i][j] + (1-mom)*lrate*LG[i][j] - (reg*dB[i][j]/M)
            j = j + 1

    return (dW, dB)

def upW(W, Bias, dW, dB, L):
    
    for i in range(L):
        W[i] = np.add(W[i], dW[i])
        Bias[i+1] = np.add(Bias[i+1], dB[i+1])
    
    return (W, Bias)