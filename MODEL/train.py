import numpy as np
import random
import inputreader as ip
import forward as fwd
import backprop as backp
import actavg as aavg
import matplotlib.pyplot as plt

f = open('files\\config.txt', "r")

L = int(f.readline())
N = int(f.readline())
Af = int(f.readline())
I = int(f.readline())
O = int(f.readline())

B = int(f.readline())
EP = int(f.readline())
lrate = float(f.readline())
mom = float(f.readline())
reg = float(f.readline())

sentence = str(L) + ": hidden layers\n" + str(N) + ": Neurons in hidden layers\n" + str(Af) + ": Activation function\n" + str(I) + ": Input Paramaters\n" + str(O) + ": Output Paramaters\n"

print("start settings :\n",sentence)

########################################################################
#_initializing_parameters
    
cfv = []
cft = []
Err = None

#_weights_random_initialization
Wf = None
W = []
W.append( np.random.rand(I, N))

for i in range(1, L):
    W.append( np.random.rand(N,N))

W.append( np.random.rand(N, O))

#_neurons_zero_initialization
An = []
An.append( np.zeros(I))

for i in range(1, L+1):
    An.append( np.zeros(N))

An.append( np.zeros(O))

#_bias_random_initialization
BiasF = None
Bias = []
Bias.append( np.zeros(I))

for i in range(1, L+1):
    Bias.append( np.random.rand(N,))

Bias.append( np.random.rand(O,))

#_delta_bias_zero_initialization
dB = []
dB.append( np.zeros(I))

for i in range(1, L+1):
    dB.append( np.zeros(N))

dB.append( np.zeros(O))


#_delta_Weights_zero_initialization
dW = []
dW.append( np.zeros((I, N)))

for i in range(1, L):
    dW.append( np.zeros((N,N)))

dW.append( np.zeros((N, O)))

########################################################################
#_processing_training_data

train_i = ip.inparse(1, I, Af, "TRAINING")
train_o = ip.inparse(2, O, Af, "TRAINING")

########################################################################
#_processing_validation_data

vali_i = ip.inparse(1, I, Af, "VALIDATION")
vali_o = ip.inparse(2, O, Af, "VALIDATION")

########################################################################
#training
pert = 0
for ep in range(EP):

    percent = 100*ep/EP
    if(percent > pert):
        print(int(percent),"%")
        pert = pert + 10

    seed = random.random()

    random.Random(seed).shuffle(train_i)
    random.Random(seed).shuffle(train_o)

    i = 0
    BAn = [An]*B
    BErr = [[0]*O]*B
    Err = [None]*O

    for v in train_i:
        
        BAn[i%B] = fwd.output(W, An, O, Af, v, L, Bias)
        BErr[i%B] = fwd.erroratout(train_o[i], BAn[i%B][L+1])
        i = i + 1    

        if(i%B == (B-1)):
            An = aavg.batchaverage(BAn, An, B)
            Err = aavg.batcherr(BErr, Err, B)
                
            (dW, dB)  = backp.backprop(dW, dB, W, Bias, L, An, Err, lrate, mom, Af, I, N, O, reg)

            (W, Bias) = backp.upW(W, Bias, dW, dB, L)

    #validation_error
    vcf = 0
    vcc = 0
    for val in vali_i:
        vAn = fwd.output(W, An, O, Af, val, L, Bias)
        vErr = fwd.erroratout(vali_o[vcc], vAn[L+1])
        vcf = vcf + np.dot(vErr, vErr)/O
        vcc = vcc + 1

    cfv.append(100*vcf/vcc)

    #training_error
    tcf = 0
    tcc = 0
    for val in train_i:
        tAn = fwd.output(W, An, O, Af, val, L, Bias)
        tErr = fwd.erroratout(train_o[tcc], tAn[L+1])
        tcf = tcf + np.dot(tErr, tErr)/O
        tcc = tcc + 1

    cft.append(100*tcf/tcc)

    if(np.min(cfv) == 100*vcf/vcc):
        Wf = W.copy()
        Biasf = Bias.copy()

########################################################################
#plotting_error_percentage
plt.title("Err Percent vs Epochs")
plt.xlabel("epoch")
plt.ylabel("error")
plt.plot(cfv,label = "validation" , color = 'red')
plt.plot(cft, label = "training" , color = 'blue')
plt.axvline(x = np.argmin(cfv), label = "validation lowest error", color = 'red',linestyle='dashed')
plt.axvline(x = np.argmin(cft), label = "training lowest error", color = 'blue',linestyle='dashed')
leg = plt.legend(loc='center right')
plt.show()


########################################################################
#saving_trained_model
wtxt = open('files\\trained.txt', "w")
wtxt.write(str(L)+"\n")
wtxt.write(str(N)+"\n")
wtxt.write(str(Af)+"\n")
wtxt.write(str(I)+"\n")
wtxt.write(str(O)+"\n")

wtxt.write("Weights:\n")
for layer in W:
    wtxt.write("[\n")
    for neu in layer:
        wtxt.write("{\n")
        for con in neu:
            wtxt.write(str(con))
            wtxt.write("\n")
        wtxt.write("}\n")
    wtxt.write("]\n")

wtxt.write("Biases:\n")
for layer in Bias:
    wtxt.write("[\n")
    for neu in layer:
        wtxt.write(str(neu))
        wtxt.write("\n")
    wtxt.write("]\n")

wtxt.close()

########################################################################
"""
print("\n\nactivations\n\n")

for x in An:
    for y in x:
        print(y, " ")
    print("\n")

print(train_o[i-1])


print("\n\nweights\n\n")

for x in W:
    for y in x:
        print(y, " ")
    print("\n")

print("\n\nbias\n\n")

for x in Bias:
    for y in x:
        print(y, " ")
    print("\n")

print("\n\nD-weights\n\n")

for x in dW:
    for y in x:
        print(y, " ")
    print("\n")

print("\n\nD-bias\n\n")

for x in dB:
    for y in x:
        print(y, " ")
    print("\n")

print("\n\nError\n\n")
for x in Err:
    print(x, " ")
print("\n")
"""

input("---end---")
