"""
Code provided by Kevin Swersky, Danny Tarlow, Ilya Sutskever, Ruslan Salakhutdinov, Rich Zemel and Ryan Adams.

Permission is granted for anyone to copy, use, modify, or distribute this
program and accompanying programs and documents for any purpose, provided
this copyright notice is retained and prominently displayed, along with
a note saying that the original programs are available from our 
web page.

The programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.

This code implements the methods described in the paper:
Cardinality Restricted Boltzmann Machines. NIPS 2012.
"""

from numpy import *
from scipy import *
from pylab import *
from matplotlib import *
import matplotlib.cm as cm
def print_aligned(w,normalize=True):
    n1 = int(ceil(sqrt(shape(w)[1])))
    n2 = n1
    r1 = int(sqrt(shape(w)[0]))
    r2 = r1
    Z = zeros(((r1+1)*n1, (r1+1)*n2), 'd')
    i1, i2 = 0, 0
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1*n2+i2
            if i>=shape(w)[1]: break
            if (normalize):
                Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = fix(w[:,i].reshape(r1,r2))
            else:
                Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
    imshow(Z,cmap=cm.gray,interpolation='nearest')
    return Z
def fix(X):
   Y = X - X.min()
   Y /= Y.max()
   return Y

def show_single(X):
   imshow(fix(X.reshape(3,10,10).transpose([1,2,0])),
interpolation='nearest')

def print_aligned_color(w,normalize=False):
    s = w.shape[0] / 3
    ss = int(sqrt(s))
    w1 = w[0:s,:]
    w2 = w[s:2*s,:]
    w3 = w[2*s:,:]
    show_batch(w1,w2,w3,normalize=normalize)

def show_batch(X1,X2,X3,normalize=False):
    Z1 = print_aligned(X1,normalize=normalize)
    Z2 = print_aligned(X2,normalize=normalize)
    Z3 = print_aligned(X3,normalize=normalize)

    img = array([Z1,Z2,Z3]).transpose([1, 2, 0])

    imshow(fix(img), interpolation='nearest')