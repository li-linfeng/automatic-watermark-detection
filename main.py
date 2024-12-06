from src import *

gx, gy, gxlist, gylist = estimate_watermark('inputs')

# est = poisson_reconstruct(gx, gy, np.zeros(gx.shape)[:,:,0])
cropped_gx, cropped_gy = crop_watermark(gx, gy)
W_m = poisson_reconstruct(cropped_gx, cropped_gy)

# random photo
img = cv2.imread('inputs/1.jpg')
im, start, end = watermark_detector(img, cropped_gx, cropped_gy)

# plt.imshow(im)
# plt.show()
# We are done with watermark estimation
# W_m is the cropped watermark
num_images = len(gxlist)

J, img_paths = get_cropped_images('inputs', num_images, start, end, cropped_gx.shape)

Wm = W_m - W_m.min()

# get threshold of W_m for alpha matte estimate
alph_est = estimate_normalized_alpha(J, Wm)
alph = np.stack([alph_est, alph_est, alph_est], axis=2)
C, est_Ik = estimate_blend_factor(J, Wm, alph)

alpha = alph.copy()
for i in range(3):
	alpha[:,:,i] = C[i]*alpha[:,:,i]

Wm = Wm + alpha*est_Ik

W = Wm.copy()
for i in range(3):
	W[:,:,i]/=C[i]

Jt = J[:25]
# now we have the values of alpha, Wm, J
# Solve for all images
# Wk, Ik, W, alpha = solve_images(Jt, W_m, alpha, W, iters=2)
# 使用 GPU 版本
Wk, Ik, W, alpha = solve_images_gpu(Jt, W_m, alpha, W, iters=2)
# W_m_threshold = (255*PlotImage(np.average(W_m, axis=2))).astype(np.uint8)
# ret, thr = cv2.threshold(W_m_threshold, 127, 255, cv2.THRESH_BINARY)  

# 创建 results 文件夹（如果不存在）
if not os.path.exists('results'):
    os.makedirs('results')

# 保存去水印后的图像
for idx, img in enumerate(Ik):
    cv2.imwrite(f'results/processed_image_{idx}.jpg', img)