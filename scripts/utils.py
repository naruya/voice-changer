import numpy as np
import matplotlib.pyplot as plt

def humming(size):
    x = np.linspace(0,size, size)
    y = 0.5 - 0.5 * np.cos(2*np.pi*x/size)
    return y
# plt.plot(humming(40))
# plt.plot(data1*humming(len(data1)))

def sigmoid(size):
    x = np.linspace(0,1,size)
    x = (x - np.mean(x))*10
    y = 1 / (1 + np.e**-x)
    return y

def connect(a, b, DEMO=False):
    w = sigmoid(40)
    c = a*np.array([(1-w)]) + b*np.array([w])
    return c[0]

# plt.figure(figsize=(6,2)); 
# plt.subplot(1,2,1); plt.plot(sigmoid(40))
# a = np.arange(0,40); b = np.ones(40)*20
# plt.subplot(1,2,2); plt.plot(a, linewidth=0.5); plt.plot(b, linewidth=0.5); plt.plot(connect(a,b)); plt.show()

def df(f):
    df = np.zeros((len(f),2))
    df[:-1] = f[1:] - f[:-1]
    return df

def denoise(f):
    v = np.ones(8)/8
    f = np.array([np.convolve(fi,v, mode='same') for fi in f.T]).T
    return f

# print("【df】")
# _x = np.zeros((1600,2))
# _x[:,0] = np.sin(np.arange(1600)/100.0) # y = sin(x/100) dy/dx = 1/100*cos(x/100)
# plt.figure(figsize=(8,1))
# plt.subplot(1,2,1); plt.plot(_x)
# plt.subplot(1,2,2); plt.plot(df(_x)); plt.show()

# print("【denoise】")
# _x = np.zeros((1600,2))
# _x[:,0] = np.sin(np.arange(1600)/100.0)
# _x += np.random.randn(1600,2)/10
# plt.figure(figsize=(12,1))
# plt.subplot(1,3,1); plt.plot(_x)
# plt.subplot(1,3,2); plt.plot(denoise(_x))
# plt.subplot(1,3,3); plt.plot(denoise(denoise(_x))); plt.show()