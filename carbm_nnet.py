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
import load_datasets as ld
import pylab
import chain_alg_wrapper as caw
import time
import sys
import os

def carbm_nnet(W, bh, O, bo, data, targets, kmin, kmax, compute_grad=False, l2=0, r=None, ca=None):
    """
    One layer feed-forward CaRBM neural network. This code may act unstable from time to time
    due to the message passing being in exp space instead of log space. Please email
    the authors if you desperately need this fixed.

    Dead units may be a problem so pretraining is recommended. KL sparsity could also be used.

    data - NxD matrix of features.
    targets - NxC matrix of class labels (encoded as 1-hot row vectors).
    W - input weights.
    bh - input biases.
    O - output weights.
    bo - output biases.
    kmin - Minimum number of active units.
    kmax - Maximum number of active units.
    compute_grad - Return the gradient if true.
    l2 - L2 penalty strength.
    r - Finite difference step length (if computing gradient).
    ca - A ChainAlg object, this is just useful to pass in if you don't want one constructed
         each time this function is called.
    """

    N,D = data.shape
    C = targets.shape[1]
    nh = W.shape[1]

    if (ca is None):
        ca = caw.ChainAlg(W.shape[1], kmin, kmax, N)

    node_potentials = np.dot(data,W)+bh
    exp_node_potentials = np.exp(node_potentials)

    #The marginals are the nonlinearities
    #The samples are useless for the neural net, but it's built into the inference code
    marginals, samples = ca.infer(exp_node_potentials, storage=0)

    #Make predictions
    total_output = np.dot(marginals,O)+bo

    #Compute the objective
    obj = np.sum(np.sum(total_output*targets,1) - nu.logsumexp(total_output,1))/N

    #L2 penalties
    obj -= 0.5*l2*np.sum(O**2)
    obj -= 0.5*l2*np.sum(W**2)
    
    if compute_grad:
        class_probs = np.exp(total_output - np.tile(nu.logsumexp(total_output,1)[:,np.newaxis],(1,C)))
        error_signal = targets - class_probs
    
        #Derivatives wrt output weights and biases
        dO = np.dot(marginals.T,error_signal)/N
        dbo = np.sum(error_signal,0)/N

        #Very accurate approximate marginals via one-sided finite-difference approximations
        #(using only one additional call to inference)
        #See "Implicit Differentiation by Perturbation", Justin Domke (NIPS 2010)
        d_loss_d_marginals = np.dot(error_signal,O.T)    

        #This step length is recommended in Domke's paper.
        if (r is None):
            eps = 2.220446049250313e-16
            r = np.sqrt(eps)*(1+np.abs(node_potentials).max())/(np.abs(d_loss_d_marginals).max())

        new_node_potentials = node_potentials + r*d_loss_d_marginals
        new_exp_node_potentials = np.exp(new_node_potentials)
        perturbed_marginals, junknew = ca.infer(new_exp_node_potentials, storage=1)
        
        dW = np.dot(data.T,(perturbed_marginals-marginals)/r)/N
        dbh = np.sum((perturbed_marginals-marginals)/r,0)/N

        #L2 penalty gradients
        dW -= l2*W
        dO -= l2*O

    #We return the negative obj and gradients since we will be doing minimization
    return -obj, {'gW':-dW, 'gbh':-dbh, 'gO':-dO, 'gbo':-dbo}

def carbm_nnet_classify(W, bh, O, bo, data, kmin, kmax, ca=None):
    N,D = data.shape
    nh = W.shape[1]

    if (ca is None):
        ca = caw.ChainAlg(W.shape[1], kmin, kmax, N)

    node_potentials = np.dot(data,W)+bh
    exp_node_potentials = np.exp(node_potentials)

    #The marginals are the nonlinearities
    #The samples are useless for the neural net, but it's built into the inference code
    marginals, samples = ca.infer(exp_node_potentials, storage=0)

    #Make predictions
    total_output = np.dot(marginals,O)+bo

    return total_output.argmax(1)

def train_carbm_nnet(data, targets, **kwargs):
    #The default settings are NOT optimally tuned, so hyperparameter tuning is required.
    init_params = kwargs.get('init_params',None)
    num_hid     = kwargs.get('num_hid',100)
    eta         = kwargs.get('eta',0.1)
    mo          = kwargs.get('mo',0)
    kmin        = kwargs.get('kmin',10)
    kmax        = kwargs.get('kmax',10)
    num_iters   = kwargs.get('num_iters',10)
    batch_size  = kwargs.get('batch_size',100)
    l2          = kwargs.get('l2',1e-5)

    if kmin is not None and kmax is not None:
        ca = caw.ChainAlg(num_hid,kmin,kmax,batch_size)

    #These are used for momentum, they are not gradients like those defined above.
    dW  = 0
    dbh = 0
    dO  = 0
    dbo = 0

    N,D = data.shape
    C = targets.shape[1]
    num_batches = np.ceil(np.double(N)/batch_size)

    if init_params is None:
        W  = 0.1*np.random.randn(D,num_hid)
        bh = np.zeros(num_hid)
        O  = 0.1*np.random.randn(num_hid,C)
        bo = np.zeros(C)
    else:
        W  = init_params['W']
        bh = init_params['bh']
        O  = init_params['O']
        bo = init_params['bo']

    for i in range(num_iters):
        obj = 0
        randIndices = np.random.permutation(N)
        iteration_obj = 0
        for batch in range(int(num_batches)):
            print 'Iteration %d batch %d of %d\r' % (i+1,batch+1,num_batches), 
            batch_X = data[randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),N)]]
            batch_y = targets[randIndices[np.mod(range(batch*batch_size,(batch+1)*batch_size),N)]]

            obj,G = carbm_nnet(W,bh,O,bo,
                             batch_X,batch_y,
                             kmin,kmax,
                             compute_grad=True,
                             l2=l2,
                             ca=ca)

            iteration_obj += obj/num_batches

            gW  = G['gW']
            gbh = G['gbh']
            gO  = G['gO']
            gbo = G['gbo']

            dW  = mo*dW - eta*gW
            dbh = mo*dbh - eta*gbh
            dO  = mo*dO - eta*gO
            dbo = mo*dbo - eta*gbo

            W = W + dW
            bh = bh + dbh
            O = O + dO
            bo = bo + dbo

        print 'Iteration %d Complete. Objective value: %s' % (i+1,iteration_obj)

    return W,bh,O,bo
if __name__ == '__main__':
    np.random.seed(1)
    X,y,Xtest,ytest = ld.load_mnist(range(10))[:4]
    kmin = 10
    kmax = 10

    W,bh,O,bo = train_carbm_nnet(X,y,mo=0.5,kmin=kmin,kmax=kmax)
    yhat = carbm_nnet_classify(W,bh,O,bo,Xtest,10,10)
    
    err = (yhat != ytest.argmax(1)).mean()
    print 'Classification error: %s' % err

    pylab.ioff()
    d.print_aligned(W)
    pylab.show(block=True)