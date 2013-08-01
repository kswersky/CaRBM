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

import nnet_utils as nu
import numpy as np
import display_filters as d
import pylab
import chain_alg_wrapper as caw
import time
import sys
import os
sys.path.append(os.getcwd() + '/data/')

class VisTypes:
    BERNOULLI = 1
    GAUSSIAN = 2

def sprbm_cd(X,**kwargs):
    print 'Training the SpRBM.'
    eta = kwargs.get('eta',0.01) #The learning rate, usually [1e-3,1].
    mo = kwargs.get('mo',0.9) #Momentum to speed up learning (can be unstable), usually [0,0.9].
    num_hid = kwargs.get('num_hid',100) #num_hid = args.nh #The number of hidden units.
    num_iters = kwargs.get('num_iters',100) #The number of epochs of learning.
    batch_size = kwargs.get('batch_size',100) #The size of each mini-batch.
    kl = kwargs.get('kl',1) #The strength of the column-wise KL penalty to prevent dead units.
    l2 = kwargs.get('l2',1e-5) #The strength of the l2 weight-decay penalty.
    sp = kwargs.get('sp',0.1) #The amount of sparsity, i.e. nh=100 and sp=0.1 means 10 hidden units will get activated.
    vistype = kwargs.get('vistype',VisTypes.BERNOULLI)
    sig = kwargs.get('sig',1)
    save_file = kwargs.get('save_file',None)
    plotting = kwargs.get('plotting',False)

    if vistype != VisTypes.GAUSSIAN:
        sig = 1

    num_on = float(np.floor(sp*num_hid))
    p = num_on / batch_size
    num_drop = num_hid - num_on
    num_examples = X.shape[0]
    num_batches = np.ceil(np.double(num_examples)/batch_size)
    ca = caw.ChainAlg(num_hid, int(num_on), int(num_on), batch_size)

    W = 0.1*np.random.randn(X.shape[1],num_hid)
    b = np.zeros(num_hid)
    c = np.zeros(X.shape[1])

    dW = 0
    db = 0
    dc = 0

    m = X.mean(0)
    times = []
    errs = []
    times.append(time.time())
    for i in range(num_iters):
        obj = 0
        randIndices = np.random.permutation(num_examples)
        for batch in range(int(num_batches)):
            print 'Iteration %d batch %d of %d\r' % (i+1,batch+1,num_batches), 
            batch_X = X[randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),num_examples)]] - m

            #POSITIVE PHASE
            node_pots_h = (np.dot(batch_X,W)/sig) + b
            mh_pos = nu.sigmoid(node_pots_h)
            h_pos = np.double(mh_pos > np.random.rand(*mh_pos.shape))
            
            #NEGATIVE PHASE
            node_pots_v = np.dot(h_pos,W.T)+c
            if vistype == VisTypes.BERNOULLI:
                mv_neg = nu.sigmoid(node_pots_v)
                v_neg = np.double(mv_neg > np.random.rand(*mv_neg.shape))
            elif vistype == VisTypes.GAUSSIAN:
                mv_neg = node_pots_v
                v_neg = np.sqrt(sig)*np.random.randn(*mv_neg.shape) + mv_neg

            v_neg = v_neg - m

            node_pots_h = (np.dot(v_neg,W)/sig) + b
            mh_neg = nu.sigmoid(node_pots_h)
            h_neg = np.double(mh_neg > np.random.rand(*mh_neg.shape))

            #Keep a weighted running average of the hidden unit activations
            if (i > 0):
                q = 0.9*q + 0.1*h_pos.mean(0)
            else:
                q = h_pos.mean(0)

            dW = mo*dW + (eta*(np.dot(batch_X.T,h_pos) - np.dot(v_neg.T,h_neg)) / (batch_size))
            db = mo*db + eta*np.mean(h_pos - h_neg,0)
            dc = mo*dc + eta*np.mean((batch_X - v_neg),0)

            dW = dW + (eta*kl*np.dot(batch_X.T,np.tile(p-q,(batch_size,1))) / batch_size)
            dW = dW - l2*W
            db = db + eta*kl*(p-q)

            W = W + dW
            b = b + db
            c = c + dc

            obj = obj + np.sum((batch_X - v_neg)**2) / (X.shape[0])
            if obj > 1e10 or not np.isfinite(obj):
                print '\nLearning has diverged.'
                if os.path.isfile(save_file):
                    print 'Deleting %s' % save_file
                    os.remove(save_file)
                    print 'File deleted: %s' % ~os.path.isfile(save_file)
                return W,b,c
        print 'Iteration %d complete. Objective value: %s' % (i+1,obj)
        times.append(time.time())
        errs.append(obj)

        #Save out some results about timing/reconstruction error per epoch.
        if save_file is not None:
            np.savez(save_file,W=W,b=b,c=c,times=times,errs=errs)

        #Plot some diagnostics for this epoch.
        if plotting:
            pylab.ion()
            pylab.subplot(1,2,1)
            d.print_aligned(W[:,0:np.minimum(100,W.shape[1])])
            pylab.subplot(1,2,2)
            d.print_aligned((v_neg + m).T)
            pylab.draw()

    return W,b,c


def carbm_cd(X,**kwargs):
    print 'Training the CaRBM.'
    eta = kwargs.get('eta',0.01) #The learning rate, usually [1e-3,1].
    mo = kwargs.get('mo',0.9) #Momentum to speed up learning (can be unstable), usually [0,0.9].
    num_hid = kwargs.get('num_hid',100) #num_hid = args.nh #The number of hidden units.
    num_iters = kwargs.get('num_iters',100) #The number of epochs of learning.
    batch_size = kwargs.get('batch_size',100) #The size of each mini-batch.
    kl = kwargs.get('kl',10) #The strength of the column-wise KL penalty to prevent dead units.
    l2 = kwargs.get('l2',1e-5) #The strength of the l2 weight-decay penalty.
    sp = kwargs.get('sp',0.1) #The amount of sparsity, i.e. nh=100 and sp=0.1 means 10 hidden units will get activated.
    vistype = kwargs.get('vistype',VisTypes.BERNOULLI)
    sig = kwargs.get('sig',1)
    save_file = kwargs.get('save_file',None)
    plotting = kwargs.get('plotting',False)

    if vistype != VisTypes.GAUSSIAN:
        sig = 1

    num_on = float(np.floor(sp*num_hid))
    p = num_on / batch_size
    num_drop = num_hid - num_on
    num_examples = X.shape[0]
    num_batches = np.ceil(np.double(num_examples)/batch_size)
    ca = caw.ChainAlg(num_hid, int(num_on), int(num_on), batch_size)

    W = 0.1*np.random.randn(X.shape[1],num_hid)
    b = np.zeros(num_hid)
    c = np.zeros(X.shape[1])

    dW = 0
    db = 0
    dc = 0

    m = X.mean(0)
    times = []
    errs = []
    times.append(time.time())
    for i in range(num_iters):
        obj = 0
        randIndices = np.random.permutation(num_examples)
        for batch in range(int(num_batches)):
            print 'Iteration %d batch %d of %d\r' % (i+1,batch+1,num_batches), 
            batch_X = X[randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),num_examples)]] - m

            #POSITIVE PHASE
            node_pots_h = (np.dot(batch_X,W)/sig) + b
            exp_node_pots_h = np.exp(node_pots_h)
            mh_pos, h_pos = ca.infer(exp_node_pots_h, storage=0)
            
            #NEGATIVE PHASE
            node_pots_v = np.dot(h_pos,W.T)+c
            if vistype == VisTypes.BERNOULLI:
                mv_neg = nu.sigmoid(node_pots_v)
                v_neg = np.double(mv_neg > np.random.rand(*mv_neg.shape))
            elif vistype == VisTypes.GAUSSIAN:
                mv_neg = node_pots_v
                v_neg = np.sqrt(sig)*np.random.randn(*mv_neg.shape) + mv_neg

            v_neg = v_neg - m

            node_pots_h = (np.dot(v_neg,W)/sig) + b
            exp_node_pots_h = np.exp(node_pots_h)
            mh_neg, h_neg = ca.infer(exp_node_pots_h, storage=1)

            #Keep a weighted running average of the hidden unit activations
            if (i > 0):
                q = 0.9*q + 0.1*h_pos.mean(0)
            else:
                q = h_pos.mean(0)

            dW = mo*dW + (eta*(np.dot(batch_X.T,h_pos) - np.dot(v_neg.T,h_neg)) / (batch_size))
            db = mo*db + eta*np.mean(h_pos - h_neg,0)
            dc = mo*dc + eta*np.mean((batch_X - v_neg),0)

            dW = dW + (eta*kl*np.dot(batch_X.T,np.tile(p-q,(batch_size,1))) / batch_size)
            dW = dW - l2*W
            db = db + eta*kl*(p-q)

            W = W + dW
            b = b + db
            c = c + dc

            obj = obj + np.sum((batch_X - v_neg)**2) / (X.shape[0])
            if obj > 1e10 or not np.isfinite(obj):
                print '\nLearning has diverged.'
                if os.path.isfile(save_file):
                    print 'Deleting %s' % save_file
                    os.remove(save_file)
                    print 'File deleted: %s' % ~os.path.isfile(save_file)
                return W,b,c
        print 'Iteration %d complete. Objective value: %s' % (i+1,obj)
        times.append(time.time())
        errs.append(obj)

        #Save out some results about timing/reconstruction error per epoch.
        if save_file is not None:
            np.savez(save_file,W=W,b=b,c=c,times=times,errs=errs)

        #Plot some diagnostics for this epoch.
        if plotting:
            pylab.ion()
            pylab.subplot(1,2,1)
            d.print_aligned(W[:,0:np.minimum(100,W.shape[1])])
            pylab.subplot(1,2,2)
            d.print_aligned((v_neg + m).T)
            pylab.draw()

    return W,b,c

if __name__ == '__main__':
    np.random.seed(1)
    import load_datasets as ld
    X = ld.load_mnist(range(10))[0]
    W,b,c = carbm_cd(X,plotting=True,num_iters=10)
    pylab.ioff()
    d.print_aligned(W)
    pylab.show(block=True)
