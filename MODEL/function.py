import numpy as np
import inputreader as ip
import forward as fwd

f = open('files\\trained.txt', "r")

L = int(f.readline())
N = int(f.readline())
Af = int(f.readline())
I = int(f.readline())
O = int(f.readline())

Burn = None

########################################################################
W = []
W.append( np.random.rand(I, N))

for i in range(1, L):
    W.append( np.random.rand(N,N))

W.append( np.random.rand(N, O))

########################################################################
Bias = []
Bias.append( np.zeros(I))

for i in range(1, L+1):
    Bias.append( np.random.rand(N,))

Bias.append( np.random.rand(O,))

########################################################################
An = []
An.append( np.zeros(I))

for i in range(1, L+1):
    An.append( np.zeros(N))

An.append( np.zeros(O))

########################################################################

i = 0
Burn = f.readline()
for layer in W:
    j = 0
    Burn = f.readline()
    for neu in layer:
        k = 0
        Burn = f.readline()
        for con in neu:
            W[i][j][k] = float(f.readline())

            k = k + 1
        j = j +1
        Burn = f.readline()
    i = i + 1
    Burn = f.readline()

i = 0
Burn = f.readline()
for layer in Bias:
    j = 0
    Burn = f.readline()
    for neu in layer:
        Bias[i][j] = float(f.readline())

        j = j +1
    i = i + 1
    Burn = f.readline()

########################################################################

val_i = ip.inparse(1, I, Af, "TEST")

for v in val_i:
    print(str(v), "->", str(fwd.output(W, An, O, Af, v, L, Bias)[L+1]), "\n")


input("---end---")