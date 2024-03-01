import numpy as np
from scipy.optimize import curve_fit


##### exp curve function
def curve_func(x, a, b, c):
    return a * (1 - np.exp(- b * x ** c))

def curve_derivation(x, a, b, c):
    # x = x + 1e-6  # numerical robustness
    return a * c * b * np.exp(-b * x ** c) * (x ** (c - 1))

def fit_curve(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 0.5, 0.5), 
                           method='trf', bounds=([0,0,0],[1,np.inf,1]))#,
                           # sigma=np.ones_like(y)*0.1, absolute_sigma=True)
                           # sigma=np.geomspace(0.5, .1, len(y)), absolute_sigma=False)
    #method='trf', sigma=np.geomspace(1, .1, len(y)), absolute_sigma=True, bounds=([0, 0, 0], [1, np.inf, 1]))
    # error = np.sqrt(np.diag(pcov))
    return tuple(popt)#, error


##### Linear function
def linear_func(x, a, b):
    # x = x + 1e-6  # numerical robustness
    return a*x+b

def fit_linear(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 0), 
                           method='trf', bounds=([0, -np.inf], [np.inf, np.inf]))
    return tuple(popt)#, error


##### numerical gradient calculation
def cal_ngs_to_dict(data, wsizes):
    n_ep = len(data)
    ngs_dict = {b:[] for b in wsizes}
    for bi, ws in enumerate(wsizes):
        if n_ep>=ws:
            # calculate numerical gradients
            x0 = np.arange(n_ep-ws+1,n_ep+1)
            y0 = np.array(data[n_ep-ws:n_ep])
            a, b = fit_linear(linear_func, x0, y0)
            ngs_dict[ws].append(a)
    return ngs_dict


##### Adaptive correction trigger module (ACT) module 
def act_module(data, ngs_dict, wsizes, detect_eps):
    '''
    Parameters
    ----------
    data : list
        Training accuracies.
    ngs_dict : dict
        Numerical gradients with different sliding window sizes.
    wsizes : list
        List of sliding window sizes used to determine whether the training has passed the minimal gradients.
    detect_eps : array
        Detected ending points of transition phase using different sliding window sizes.

    Return
    -------
    detected correction trigger point

    '''
    check_buff = np.mean(wsizes)
    n_ep = len(data)
    
    # calculate numerical gradients
    for bi, ws in enumerate(wsizes):
        if n_ep>=ws:
            # calculate numerical gradients
            x0 = np.arange(n_ep-ws+1,n_ep+1)
            y0 = np.array(data[n_ep-ws:n_ep])
            a, b = fit_linear(linear_func, x0, y0)
            ngs_dict[ws].append(a)
    
            # check whether gradients start to decrease
            if min(ngs_dict[ws])<a:
                ind = np.argmin(ngs_dict[ws])+ws
                if n_ep-ind > check_buff:
                    detect_eps[bi] = ind
    
    # output final detection result
    if (detect_eps>0).sum()==len(wsizes):
        # training has reached the end of transition stage
        dep = int(np.mean(detect_eps))  # final detected ending point of transition stage
        
        # fitting training accuracies (y) to exponential function
        x0 = np.arange(dep)+1
        y0 = np.array(data[:dep])
        a, b, c = fit_curve(curve_func, x0, y0)
        
        # estimated y using fitted function
        yh = curve_func(x0, a, b, c)
        
        # adaptive threshold
        thr = (yh[-1]-yh[0])/(x0[-1]-x0[0])
        
        # gradients of y (fitted function)
        yd = curve_derivation(x0, a, b, c)
        
        # transition phase starting point
        mid = np.sum(yd>thr)
        
        # final detected trigger point
        fdep = int((dep+mid)/2)
    else:
        fdep = 0

    return fdep, ngs_dict, detect_eps
    