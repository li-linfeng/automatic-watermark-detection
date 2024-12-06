from __future__ import division

import numpy as np
import scipy.sparse
import scipy
from scipy.sparse import *
from numpy.lib.stride_tricks import as_strided


def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


# Returns sparse matting laplacian
def computeLaplacian(img, eps=10**(-7), win_rad=1):
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2*win_rad, w - 2*win_rad
    win_diam = win_rad*2+1

    indsM = np.arange(h*w).reshape((h, w))
    ravelImg = img.reshape(h*w, d)
    win_inds = rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=2, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI)/win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L


def closed_form_matte(img, scribbled_img, mylambda=100):
    h, w, c = img.shape
    consts_map = (np.sum(abs(img - scribbled_img), axis=-1)>0.001).astype(np.float64)
    consts_vals = scribbled_img[:,:,0]*consts_map
    D_s = consts_map.ravel()
    b_s = consts_vals.ravel()
    
    print("Computing Matting Laplacian")
    L = computeLaplacian(img)
    print("Laplacian shape:", L.shape)
    
    sD_s = scipy.sparse.diags(D_s)
    A = L + mylambda*sD_s
    b = mylambda*b_s
    
    print("Starting iterative solver...")
    try:
        # 使用迭代求解器 bicgstab (双共轭梯度稳定法)
        x, info = scipy.sparse.linalg.bicgstab(A, b, 
                                              tol=1e-4,    # 容差
                                              maxiter=1000) # 最大迭代次数
        if info == 0:
            print("Linear system solved successfully")
        else:
            print(f"Warning: Solver did not converge, info={info}")
    except Exception as e:
        print(f"Error solving linear system: {str(e)}")
        raise
        
    print("Reshaping solution...")
    alpha = np.minimum(np.maximum(x.reshape(h, w), 0), 1)
    print("Alpha computation completed")
    return alpha
