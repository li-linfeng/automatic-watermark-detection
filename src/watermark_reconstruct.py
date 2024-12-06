import numpy as np
import cv2
import os
import scipy
from scipy.sparse import *
from scipy.sparse import linalg
from .estimate_watermark import *
from .closed_form_matting import *
from numpy import nan, isnan

def get_cropped_images(foldername, num_images, start, end, shape):
    '''
    This is the part where we get all the images, extract their parts, and then add it to our matrix
    '''
    images_cropped = np.zeros((num_images,) + shape)
    # get images
    # Store all the watermarked images
    # start, and end are already stored
    # just crop and store image
    image_paths = []
    _s, _e = start, end
    index = 0

    # Iterate over all images
    for r, dirs, files in os.walk(foldername):

        for file in files:
            _img = cv2.imread(os.sep.join([r, file]))
            if _img is not None:
                # estimate the watermark part
                image_paths.append(os.sep.join([r, file]))
                _img = _img[_s[0]:(_s[0]+_e[0]), _s[1]:(_s[1]+_e[1]), :]
                # add to list images
                images_cropped[index, :, :, :] = _img
                index+=1
            else:
                print("%s not found."%(file))

    return (images_cropped, image_paths)


# get sobel coordinates for y
def _get_ysobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i-1, j, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i+1, j, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# get sobel coordinates for x
def _get_xsobel_coord(coord, shape):
    i, j, k = coord
    m, n, p = shape
    return [
        (i, j-1, k, -2), (i-1, j-1, k, -1), (i-1, j+1, k, -1),
        (i, j+1, k,  2), (i+1, j-1, k,  1), (i+1, j+1, k,  1)
    ]

# filter
def _filter_list_item(coord, shape):
    i, j, k, v = coord
    m, n, p = shape
    if i>=0 and i<m and j>=0 and j<n:
        return True

# Change to ravel index
# also filters the wrong guys
def _change_to_ravel_index(li, shape):
    li = filter(lambda x: _filter_list_item(x, shape), li)
    i, j, k, v = zip(*li)
    return zip(np.ravel_multi_index((i, j, k), shape), v)

# TODO: Consider wrap around of indices to remove the edge at the end of sobel
# get Sobel sparse matrix for Y
def get_ySobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_ysobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))


# get Sobel sparse matrix for X
def get_xSobel_matrix(m, n, p):
    size = m*n*p
    shape = (m, n, p)
    i, j, k = np.unravel_index(np.arange(size), (m, n, p))
    ijk = zip(list(i), list(j), list(k))
    ijk_nbrs = map(lambda x: _get_xsobel_coord(x, shape), ijk)
    ijk_nbrs_to_index = map(lambda l: _change_to_ravel_index(l, shape), ijk_nbrs)
    # we get a list of idx, values for a particular idx
    # we have got the complete list now, map it to actual index
    actual_map = []
    for i, list_of_coords in enumerate(ijk_nbrs_to_index):
        for coord in list_of_coords:
            actual_map.append((i, coord[0], coord[1]))

    i, j, vals = zip(*actual_map)
    return coo_matrix((vals, (i, j)), shape=(size, size))

# get estimated normalized alpha matte
def estimate_normalized_alpha(J, W_m, num_images=30, threshold=170, invert=False, adaptive=False, adaptive_threshold=21, c2=10):
    # 确保不超过实际图片数量
    num_images = min(num_images, len(J))
    
    num, m, n, p = J.shape
    alpha = np.zeros((num_images, m, n))
    
    _Wm = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
    if adaptive:
        thr = cv2.adaptiveThreshold(_Wm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adaptive_threshold, c2)
    else:
        ret, thr = cv2.threshold(_Wm, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        thr = 255-thr
    thr = np.stack([thr, thr, thr], axis=2)

    print("Estimating normalized alpha using %d images."%(num_images))
    # for all images, calculate alpha
    for idx in range(num_images):
        print("Estimating normalized alpha idx:%d "%(idx))
        imgcopy = thr
        alph = closed_form_matte(J[idx], imgcopy)
        alpha[idx] = alph

    # 计算所有图片的 alpha 的中值
    alpha = np.median(alpha, axis=0)
    return alpha

def estimate_blend_factor(J, W_m, alph, threshold=0.01*255):
    K, m, n, p = J.shape
    Jm = (J - W_m)
    gx_jm = np.zeros(J.shape)
    gy_jm = np.zeros(J.shape)

    for i in range(K):
        gx_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 1, 0, 3)
        gy_jm[i] = cv2.Sobel(Jm[i], cv2.CV_64F, 0, 1, 3)

    Jm_grad = np.sqrt(gx_jm**2 + gy_jm**2)

    est_Ik = alph*np.median(J, axis=0)
    gx_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 1, 0, 3)
    gy_estIk = cv2.Sobel(est_Ik, cv2.CV_64F, 0, 1, 3)
    estIk_grad = np.sqrt(gx_estIk**2 + gy_estIk**2)

    C = []
    for i in range(3):
        c_i = np.sum(Jm_grad[:,:,:,i]*estIk_grad[:,:,i])/np.sum(np.square(estIk_grad[:,:,i]))/K
        print(c_i)
        C.append(c_i)

    return C, est_Ik


def Func_Phi(X, epsilon=1e-3):
    return np.sqrt(X + epsilon**2)

def Func_Phi_deriv(X, epsilon=1e-3):
    return 0.5/Func_Phi(X, epsilon)

def solve_images(J, W_m, alpha, W_init, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=4):
    '''
    Master solver, follows the algorithm given in the supplementary.
    W_init: Initial value of W
    Step 1: Image Watermark decomposition
    '''
    # prepare variables
    K, m, n, p = J.shape
    size = m*n*p

    # 预计算 Sobel 矩阵，避免重复计算
    print("Precomputing Sobel matrices...")
    sobelx = get_xSobel_matrix(m, n, p)
    sobely = get_ySobel_matrix(m, n, p)
    
    # 初始化变量
    Ik = np.zeros(J.shape)
    Wk = np.zeros(J.shape)
    for i in range(K):
        Ik[i] = J[i] - W_m
        Wk[i] = W_init.copy()

    W = W_init.copy()

    # 预计算一些常量
    alpha_gx = cv2.Sobel(alpha, cv2.CV_64F, 1, 0, 3)
    alpha_gy = cv2.Sobel(alpha, cv2.CV_64F, 0, 1, 3)
    Wm_gx = cv2.Sobel(W_m, cv2.CV_64F, 1, 0, 3)
    Wm_gy = cv2.Sobel(W_m, cv2.CV_64F, 0, 1, 3)

    # 主迭代循环
    for iter_num in range(iters):
        print(f"\nIteration: {iter_num+1}/{iters}")

        # Step 1: Image Watermark decomposition
        print("Step 1: Image Watermark decomposition")
        cx = diags(np.abs(alpha_gx).reshape(-1))
        cy = diags(np.abs(alpha_gy).reshape(-1))
        alpha_diag = diags(alpha.reshape(-1))
        alpha_bar_diag = diags((1-alpha).reshape(-1))

        for i in range(K):
            print(f"Processing image {i+1}/{K}", end='\r')
            
            # 使用向量化操作计算梯度
            Wkx = cv2.Sobel(Wk[i], cv2.CV_64F, 1, 0, 3)
            Wky = cv2.Sobel(Wk[i], cv2.CV_64F, 0, 1, 3)
            Ikx = cv2.Sobel(Ik[i], cv2.CV_64F, 1, 0, 3)
            Iky = cv2.Sobel(Ik[i], cv2.CV_64F, 0, 1, 3)

            # 优化矩阵计算
            alphaWk = alpha*Wk[i]
            alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
            alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)

            # 使用向量化操作计算 phi 函数
            phi_data = diags(Func_Phi_deriv(np.square(alpha*Wk[i] + (1-alpha)*Ik[i] - J[i]).reshape(-1)))
            phi_W = diags(Func_Phi_deriv(np.square(np.abs(alpha_gx)*Wkx + np.abs(alpha_gy)*Wky).reshape(-1)))
            phi_I = diags(Func_Phi_deriv(np.square(np.abs(alpha_gx)*Ikx + np.abs(alpha_gy)*Iky).reshape(-1)))
            phi_f = diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2).reshape(-1)))
            phi_aux = diags(Func_Phi_deriv(np.square(Wk[i] - W).reshape(-1)))
            
            # 使用稀疏矩阵运算
            try:
                A = vstack([
                    hstack([(alpha_diag**2)*phi_data + lambda_w*L_w + beta*A_f, alpha_diag*alpha_bar_diag*phi_data]),
                    hstack([alpha_diag*alpha_bar_diag*phi_data, (alpha_bar_diag**2)*phi_data + lambda_i*L_i])
                ]).tocsr()
                
                b = np.hstack([bW, bI])
                x = linalg.spsolve(A, b, use_umfpack=False)
                
                Wk[i] = x[:size].reshape(m, n, p)
                Ik[i] = x[size:].reshape(m, n, p)
            except Exception as e:
                print(f"\nWarning: Error in solving linear system for image {i}: {e}")
                continue

        # Step 2: Update W
        print("\nStep 2: Updating W")
        W = np.median(Wk, axis=0)

        # Step 3: Update alpha
        print("Step 3: Updating alpha")
        W_diag = diags(W.reshape(-1))
        A1 = None
        b1 = None

        for i in range(K):
            print(f"Processing image {i+1}/{K} for alpha update", end='\r')
            try:
                alphaWk = alpha*Wk[i]
                alphaWk_gx = cv2.Sobel(alphaWk, cv2.CV_64F, 1, 0, 3)
                alphaWk_gy = cv2.Sobel(alphaWk, cv2.CV_64F, 0, 1, 3)
                
                # 使用向量化操作
                phi_f = diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2).reshape(-1)))
                phi_kA = diags(((Func_Phi_deriv((((alpha*Wk[i] + (1-alpha)*Ik[i] - J[i])**2)))) * ((W-Ik[i])**2)).reshape(-1))
                phi_kB = ((Func_Phi_deriv((((alpha*Wk[i] + (1-alpha)*Ik[i] - J[i])**2))))*(W-Ik[i])*(J[i]-Ik[i])).reshape(-1)
                
                if A1 is None:
                    A1 = phi_kA + lambda_a*L_alpha + beta*A_tilde_f
                    b1 = phi_kB + beta*W_diag.dot(L_f).dot(W_m.reshape(-1))
                else:
                    A1 += (phi_kA + lambda_a*L_alpha + beta*A_tilde_f)
                    b1 += (phi_kB + beta*W_diag.T.dot(L_f).dot(W_m.reshape(-1)))
            except Exception as e:
                print(f"\nWarning: Error in alpha update for image {i}: {e}")
                continue

        try:
            alpha = linalg.spsolve(A1, b1, use_umfpack=False).reshape(m, n, p)
            alpha = np.clip(alpha, 0, 1)  # 确保 alpha 在 [0,1] 范围内
        except Exception as e:
            print(f"\nWarning: Error in final alpha solve: {e}")
    
    return (Wk, Ik, W, alpha)


def changeContrastImage(J, I):
    cJ1 = J[0, 0, :]
    cJ2 = J[-1, -1, :]

    cI1 = I[0, 0, :]
    cI2 = I[-1,-1, :]

    I_m = cJ1 + (I-cI1)/(cI2-cI1)*(cJ2-cJ1)
    return I_m

def solve_images_gpu(J, W_m, alpha, W_init, gamma=1, beta=1, lambda_w=0.005, lambda_i=1, lambda_a=0.01, iters=4):
    '''
    GPU accelerated version of solve_images using CuPy
    '''
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse import diags as cp_diags
    from cupyx.scipy.sparse import vstack as cp_vstack
    from cupyx.scipy.sparse import hstack as cp_hstack
    
    # 准备变量并转移到 GPU
    K, m, n, p = J.shape
    size = m*n*p
    
    # 将数据转移到 GPU
    J_gpu = cp.array(J)
    W_m_gpu = cp.array(W_m)
    alpha_gpu = cp.array(alpha)
    W_init_gpu = cp.array(W_init)
    
    print("Precomputing Sobel matrices...")
    sobelx = get_xSobel_matrix(m, n, p)
    sobely = get_ySobel_matrix(m, n, p)
    sobelx_gpu = cp_csr_matrix(sobelx)
    sobely_gpu = cp_csr_matrix(sobely)
    
    # 初始化变量
    Ik_gpu = cp.zeros(J.shape)
    Wk_gpu = cp.zeros(J.shape)
    for i in range(K):
        Ik_gpu[i] = J_gpu[i] - W_m_gpu
        Wk_gpu[i] = W_init_gpu.copy()

    W_gpu = W_init_gpu.copy()

    # 预计算梯度
    alpha_gx = cp.array(cv2.Sobel(cp.asnumpy(alpha_gpu), cv2.CV_64F, 1, 0, 3))
    alpha_gy = cp.array(cv2.Sobel(cp.asnumpy(alpha_gpu), cv2.CV_64F, 0, 1, 3))
    Wm_gx = cp.array(cv2.Sobel(cp.asnumpy(W_m_gpu), cv2.CV_64F, 1, 0, 3))
    Wm_gy = cp.array(cv2.Sobel(cp.asnumpy(W_m_gpu), cv2.CV_64F, 0, 1, 3))

    for iter_num in range(iters):
        print(f"\nIteration: {iter_num+1}/{iters}")
        
        # Step 1: Image Watermark decomposition
        print("Step 1: Image Watermark decomposition")
        cx = cp_diags(cp.abs(alpha_gx).reshape(-1))
        cy = cp_diags(cp.abs(alpha_gy).reshape(-1))
        alpha_diag = cp_diags(alpha_gpu.reshape(-1))
        alpha_bar_diag = cp_diags((1-alpha_gpu).reshape(-1))

        for i in range(K):
            print(f"Processing image {i+1}/{K}", end='\r')
            
            # GPU 计算梯度
            Wk_cpu = cp.asnumpy(Wk_gpu[i])
            Ik_cpu = cp.asnumpy(Ik_gpu[i])
            
            Wkx = cp.array(cv2.Sobel(Wk_cpu, cv2.CV_64F, 1, 0, 3))
            Wky = cp.array(cv2.Sobel(Wk_cpu, cv2.CV_64F, 0, 1, 3))
            Ikx = cp.array(cv2.Sobel(Ik_cpu, cv2.CV_64F, 1, 0, 3))
            Iky = cp.array(cv2.Sobel(Ik_cpu, cv2.CV_64F, 0, 1, 3))

            alphaWk = alpha_gpu * Wk_gpu[i]
            alphaWk_cpu = cp.asnumpy(alphaWk)
            alphaWk_gx = cp.array(cv2.Sobel(alphaWk_cpu, cv2.CV_64F, 1, 0, 3))
            alphaWk_gy = cp.array(cv2.Sobel(alphaWk_cpu, cv2.CV_64F, 0, 1, 3))

            try:
                # GPU 矩阵运算
                phi_data = cp_diags(Func_Phi_deriv(cp.square(alpha_gpu*Wk_gpu[i] + (1-alpha_gpu)*Ik_gpu[i] - J_gpu[i]).reshape(-1)))
                phi_W = cp_diags(Func_Phi_deriv(cp.square(cp.abs(alpha_gx)*Wkx + cp.abs(alpha_gy)*Wky).reshape(-1)))
                phi_I = cp_diags(Func_Phi_deriv(cp.square(cp.abs(alpha_gx)*Ikx + cp.abs(alpha_gy)*Iky).reshape(-1)))
                phi_f = cp_diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2).reshape(-1)))
                phi_aux = cp_diags(Func_Phi_deriv(cp.square(Wk_gpu[i] - W_gpu).reshape(-1)))

                # 使用 GPU 上的 Sobel 矩阵
                L_i = sobelx_gpu.T.dot(cx*phi_I).dot(sobelx_gpu) + sobely_gpu.T.dot(cy*phi_I).dot(sobely_gpu)
                L_w = sobelx_gpu.T.dot(cx*phi_W).dot(sobelx_gpu) + sobely_gpu.T.dot(cy*phi_W).dot(sobely_gpu)
                L_f = sobelx_gpu.T.dot(phi_f).dot(sobelx_gpu) + sobely_gpu.T.dot(phi_f).dot(sobely_gpu)
                A_f = alpha_diag.T.dot(L_f).dot(alpha_diag) + gamma*phi_aux

                bW = alpha_diag.dot(phi_data).dot(J_gpu[i].reshape(-1)) + beta*L_f.dot(W_m_gpu.reshape(-1)) + gamma*phi_aux.dot(W_gpu.reshape(-1))
                bI = alpha_bar_diag.dot(phi_data).dot(J_gpu[i].reshape(-1))

                # 构建并求解线性系统
                A_gpu = cp_vstack([
                    cp_hstack([(alpha_diag**2)*phi_data + lambda_w*L_w + beta*A_f, alpha_diag*alpha_bar_diag*phi_data]),
                    cp_hstack([alpha_diag*alpha_bar_diag*phi_data, (alpha_bar_diag**2)*phi_data + lambda_i*L_i])
                ]).tocsr()
                
                b_gpu = cp.hstack([bW, bI])
                x_gpu = cp.sparse.linalg.spsolve(A_gpu, b_gpu)
                
                Wk_gpu[i] = x_gpu[:size].reshape(m, n, p)
                Ik_gpu[i] = x_gpu[size:].reshape(m, n, p)
                
            except Exception as e:
                print(f"\nWarning: Error in solving linear system for image {i}: {e}")
                continue

        # Step 2: Update W
        print("\nStep 2: Updating W")
        W_gpu = cp.median(Wk_gpu, axis=0)

        # Step 3: Update alpha
        print("Step 3: Updating alpha")
        W_diag = cp_diags(W_gpu.reshape(-1))
        A1 = None
        b1 = None

        for i in range(K):
            print(f"Processing image {i+1}/{K} for alpha update", end='\r')
            try:
                alphaWk = alpha_gpu*Wk_gpu[i]
                alphaWk_cpu = cp.asnumpy(alphaWk)
                alphaWk_gx = cp.array(cv2.Sobel(alphaWk_cpu, cv2.CV_64F, 1, 0, 3))
                alphaWk_gy = cp.array(cv2.Sobel(alphaWk_cpu, cv2.CV_64F, 0, 1, 3))
                
                phi_f = cp_diags(Func_Phi_deriv(((Wm_gx - alphaWk_gx)**2 + (Wm_gy - alphaWk_gy)**2).reshape(-1)))
                phi_alpha = cp_diags(Func_Phi_deriv(alpha_gx**2 + alpha_gy**2).reshape(-1))
                L_alpha = sobelx_gpu.T.dot(phi_alpha).dot(sobelx_gpu) + sobely_gpu.T.dot(phi_alpha).dot(sobely_gpu)
                
                L_f = sobelx_gpu.T.dot(phi_f).dot(sobelx_gpu) + sobely_gpu.T.dot(phi_f).dot(sobely_gpu)
                A_tilde_f = W_diag.T.dot(L_f).dot(W_diag)
                
                phi_kA = cp_diags(((Func_Phi_deriv((((alpha_gpu*Wk_gpu[i] + (1-alpha_gpu)*Ik_gpu[i] - J_gpu[i])**2)))) * ((W_gpu-Ik_gpu[i])**2)).reshape(-1))
                phi_kB = ((Func_Phi_deriv((((alpha_gpu*Wk_gpu[i] + (1-alpha_gpu)*Ik_gpu[i] - J_gpu[i])**2))))*(W_gpu-Ik_gpu[i])*(J_gpu[i]-Ik_gpu[i])).reshape(-1)
                
                if A1 is None:
                    A1 = phi_kA + lambda_a*L_alpha + beta*A_tilde_f
                    b1 = phi_kB + beta*W_diag.dot(L_f).dot(W_m_gpu.reshape(-1))
                else:
                    A1 += (phi_kA + lambda_a*L_alpha + beta*A_tilde_f)
                    b1 += (phi_kB + beta*W_diag.T.dot(L_f).dot(W_m_gpu.reshape(-1)))
            except Exception as e:
                print(f"\nWarning: Error in alpha update for image {i}: {e}")
                continue

        try:
            alpha_gpu = cp.sparse.linalg.spsolve(A1, b1).reshape(m, n, p)
            alpha_gpu = cp.clip(alpha_gpu, 0, 1)
        except Exception as e:
            print(f"\nWarning: Error in final alpha solve: {e}")

    # 将结果转回 CPU
    Wk = cp.asnumpy(Wk_gpu)
    Ik = cp.asnumpy(Ik_gpu)
    W = cp.asnumpy(W_gpu)
    alpha = cp.asnumpy(alpha_gpu)
    
    return (Wk, Ik, W, alpha)
