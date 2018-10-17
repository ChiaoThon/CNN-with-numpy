#verify the cnn according to https://grzegorzgwardys.wordpress.com/2016/04/22/8/

import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import time
import mnist
from scipy import signal

learn_rate = 0.0005  # learn rate
ni = 720  # dimension of FCNN input layer
nh = 100  # node count of FCNN hidden layer
no = 10  # dimension of FCNN output layer

CK = no  # number of different classes
np.random.seed(0)
wih = np.random.normal(loc=0, scale=np.sqrt(2.0 / 100), size=(ni, nh))  # weight matrix of input layer hidden layer
who = np.random.normal(loc=0, scale=np.sqrt(2.0 / 100), size=(nh, no))  # weight matrix of hidden layer tooutput layer
bih = np.random.random((1, nh))  # bias matrix of input layer hidden layer
bho = np.random.random((1, no))  # bias matrix of hidden layer tooutput layer

#size of the input
cmat_s = [28, 28]

#number of kernel
kernel_c = 5

#size of the kernel  in the form of (kernel number,kernel size)
kernel_s = [kernel_c, 5, 5]

#size of output after conv layer
cout_s = [kernel_c, 24, 24]

#pooling size
pool_s = 2

#size of output after pooling layer
pout_s = [kernel_c, 12, 12]

conv_kernel = np.random.normal(loc=0, scale=np.sqrt(2.0 / 100), size=kernel_s)

conv_bias = np.random.normal(loc=0, scale=1.0, size=(kernel_c,))

def MatRot180(k):
    k = np.rot90(k)
    return np.rot90(k)

# Convolution operation
# m:matrix 1
# k: convoluteion kernel
#mode:'full','valid','same',refer to https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.convolve2d.html
def Convolution(m, k,mode):
    return signal.convolve2d(m,k,mode)

# pooling (meanpooling)
def DownSampling(m, scale=2):
    w, h = m.shape
    rw, rh = int((w + 1) / 2), int((h + 1) / 2)
    r = [[np.mean(m[i * scale:i * scale + scale, j * scale:j * scale + scale]) for j in range(rw)] for i in range(rh)]
    return np.array(r)

#towards the meanpooling
def UpSampling(m, scale=2):
    b = np.ones((scale, scale))/(1.*scale*scale)
    return np.kron(m, b)


def Mat2Vec(m):
    m = m.reshape(1, (m.size))
    return m


def Vec2Mat(m):
    m = m.reshape(pout_s)
    return m


# m:matrix
# conv_kernel : convoluteion kernel
def RunConvLay(m):
    cout = [Convolution(m, k,'valid') for k in conv_kernel]
    return np.array(cout)


def ReLU(m):
    m = np.maximum(m, 0.0)
    return m


def DReLU(m):
    m[m > 0.0] = 1.0
    #m[m <= 0.0] = 0.0
    return m


# FCNN
def RunFcLay(x):
    xij = x.dot(wih) + bih
    hout = ReLU(xij)  # ReLU active function
    xjk = hout.dot(who) + bho
    exp_xjk = np.exp(xjk)  # softmax active function [partion]
    oout = exp_xjk / np.sum(exp_xjk, axis=1)  # softmax [partion]
    return hout, oout


# Run CNN
def RunCNN(m):
    cout = RunConvLay(m)  # convolution layer

    bout = np.array([cout[ind] + conv_bias[ind] for ind in range(kernel_c)])  # add bias	
	
    pout = np.array([DownSampling(bout[ind], pool_s) for ind in range(kernel_c)])  # pooling layer

    rout = np.array([ReLU(pout[ind]) for ind in range(kernel_c)])  # ReLU
    #print('Relu is ',rout)
    iout = Mat2Vec(rout)  # matrix to vector,flatten the features into the vector
    hout, oout = RunFcLay(iout)  # FCNN
    #print('the output is ',oout)
    return m, cout, pout, bout, rout, iout, hout, oout


# run CNN model to judge label,in other words, do the prediction
def UseCNNJudgeType(m):
    _, _, _, _, _, _, _, oout = RunCNN(m)
    return oout.argmax()


# update weight by BPNN algorithm
# cout : convolution layer output
# pout : pooling layer output
# bout : after add bias
# rout : ReLU function output
# iout : matrix to vector / FCNN input
# hout : FCNN hidden layer output
# oout : FCNN output layer output
# tarout : target output
def BackProp(m, cout, pout, bout, rout, iout, hout, oout, tarout):
    global wih, bih, who, bho, conv_bias, conv_kernel
    odelta = oout
    odelta[0, tarout] -= 1  # 1,10,only one input

    dwho = hout.T.dot(odelta)
	
    #each neuron has a bias
    dbho = np.sum(odelta, axis=0, keepdims=True)

    #the derivative of ReLu function
    dReLU = DReLU(hout)  # 1,100

    hdelta = odelta.dot(who.T) * dReLU  # 1,100
    dbih = np.sum(hdelta, axis=0)
    dwih = iout.T.dot(hdelta)

    vdelta = hdelta.dot(wih.T)  # 1,720

    #update the weights and bias in dense part
    wih -= learn_rate * dwih
    bih -= learn_rate * dbih
    who -= learn_rate * dwho
    bho -= learn_rate * dbho

    rdelta = Vec2Mat(vdelta)  # 5,12,12
    #print(rdelta.shape)
    #print(rdelta[0])    


    dReLU = np.array([DReLU(item) for item in rout])
    #print(dReLU.shape)
    #print(dReLU[0])


    #bdelta = np.array([rdelta[item]*dReLU[item] for item in range(kernel_c)])
    bdelta = rdelta * dReLU

    cdelta = np.array([UpSampling(item) for item in bdelta])

    bidelta = cdelta.sum(axis=(1, 2))  # 5,

    #print(cdelta.shape)
    #print(cout.shape) 

    dconvbias = bidelta
    dconvkernel = np.array([Convolution(kernel, MatRot180(m),'valid') for kernel in cdelta])  # 5,5,5

    conv_bias -= learn_rate * dconvbias
    conv_kernel -=learn_rate * dconvkernel


# get c numbers of [s,e] interval
def RandomList(s, e, c):
    list = range(s, e)
    slice = random.sample(list, c)
    return slice


# training CNN
# batchsize : SGD batch size
def Training(train_data, train_label):
    for j in range(len(train_data)):
        xdata = train_data[j]
        ylabel = train_label[j]
        xdata = xdata.reshape(cmat_s)
        mat, cout, pout, bout, rout, iout, hout, oout = RunCNN(xdata)
        BackProp(mat, cout, pout, bout, rout, iout, hout, oout, ylabel)
        


# test CNN model
def Testing(testing_data, testing_label):
    error_c = 0

    for i in range(len(testing_data)):
        xdata = testing_data[i]
        xdata = xdata.reshape(cmat_s)
        modeloutput = UseCNNJudgeType(xdata)
        if (modeloutput != testing_label[i]):
            error_c += 1

    return 1.0-(1.*error_c / len(testing_data))


# write weight matrix to file
# path : file path
def WriteWeight(path):
    file = open(path, 'w')
    wigstring = ""
    np.set_printoptions(threshold=np.NaN, precision=10, suppress=True)

    wigstring += 'conv_kernel\n'
    wigstring += str(conv_kernel)
    wigstring += '\n\n\n'

    wigstring += 'conv_bias\n'
    wigstring += str(conv_bias)
    wigstring += '\n\n\n'

    wigstring += 'wih\n'
    wigstring += str(wih)
    wigstring += '\n\n\n'

    wigstring += 'bih\n'
    wigstring += str(bih)
    wigstring += '\n\n\n'

    wigstring += 'who\n'
    wigstring += str(who)
    wigstring += '\n\n\n'

    wigstring += 'bho\n'
    wigstring += str(bho)
    wigstring += '\n\n\n'

    file.write(wigstring)
    file.close()



train_data = mnist.train_images()
train_label = mnist.train_labels()
testing_data = mnist.test_images()
testing_label = mnist.test_labels()

train_data=1./train_data.max()*(train_data-train_data.mean())
testing_data=1./testing_data.max()*(testing_data-testing_data.mean())



start =time.time()  # get current time
train_accu=[]
epoch_time=1
batch_size=250
for i in range(epoch_time):
    arr=np.arange(50000)
    #print(arr)

    np.random.shuffle(arr)

    #rl=RandomList(0, 100, 100)
    for j in range(int(50000/batch_size)):
        #print(arr)
        inx=arr[j*batch_size:(j*batch_size+batch_size)]
        train_da=train_data[inx,:]
        train_la=train_label[inx]
        Training(train_da, train_la)
        tr_acc=Testing(train_da,train_la) 
        train_accu.append(tr_acc)   
        print('train epo: ' + str(i+1) + '/'+str(epoch_time)+'   train batch: '+str(j+1)+'/'+str(int(50000/batch_size))+'   Train accuracy of '+str(j+1)+' st'+' batch is ', tr_acc)


print("do the test now: ")
tacc=Testing(testing_data, testing_label)
print('Test accuracy:', tacc,'\n')

finish = time.time()  # get current time

print('Elapsed Time: ', finish - start, 's')

#WriteWeight('weight.txt')
plt.plot(train_accu)
plt.show()




