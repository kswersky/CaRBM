from scipy.io import loadmat
from numpy import *
import pickle
import copy
import glob
import os
import sys
DATA_PREFIX = 'data/'
def load_cifar10_patches():
    D = loadmat(DATA_PREFIX + 'patches_16x16.mat')
    return D['patches']
    
def load_mnist(digitsRange,binary=True):
	digitsRange = copy.copy(digitsRange)
	digits = loadmat(DATA_PREFIX + 'mnist_all.mat')
	d = digits['train'+str(digitsRange[0])]
	dTest = digits['test'+str(digitsRange[0])]
	if (len(digitsRange)<=2 and binary):
		targets = zeros((d.shape[0],))
		targetsTest = zeros((dTest.shape[0],))
	else:
		targets = zeros((d.shape[0],len(digitsRange)))
		targets[0:,0] = 1
		targetsTest = zeros((dTest.shape[0],len(digitsRange)))
		targetsTest[0:,0] = 1
	digitsRange.remove(digitsRange[0])
	dimIndex = 1
	for i in digitsRange:
		d = vstack((d,digits['train'+str(i)]))
		dTest = vstack((dTest,digits['test'+str(i)]))
		index = targets.shape[0]
		indexTest = targetsTest.shape[0]
		if (len(digitsRange) <= 1 and binary):
			targets = hstack((targets,ones((digits['train'+str(i)].shape[0],))))
			targetsTest = hstack((targetsTest,ones((digits['test'+str(i)].shape[0]))))
		else:
			targets = vstack((targets,zeros((digits['train'+str(i)].shape[0],len(digitsRange)+1))))
			targets[index:,dimIndex] = 1
			
			targetsTest = vstack((targetsTest,zeros((digits['test'+str(i)].shape[0],len(digitsRange)+1))))
			targetsTest[indexTest:,dimIndex] = 1
			dimIndex = dimIndex + 1
	d = double(d)/255
	dTest = double(dTest)/255
	return d,targets,dTest,targetsTest,dTest,targetsTest