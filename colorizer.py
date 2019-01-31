import cv2
import numpy as np
from regression import gradient_decent, grayscale

input_file = r"C:\Users\nickp\OneDrive\Documents\class\cs520\Assignment 4\assignment4\img.jpeg"
gray_img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
color_img = cv2.imread(input_file, cv2.IMREAD_COLOR)

X = np.ones((gray_img.size, 10), dtype=np.float64)
yr, yg, yb = [], [], []
count = 0

# preprocess
for i in range(color_img.shape[0]):
	for j in range(color_img.shape[1]):

		points = []

		# get data from surrounding pixels
		SURROUNDING = [
			(i-1, j-1), (i-1, j), (i-1,j+1),
			(i, j-1), (i, j), (i,j+1), 
			(i+1, j-1), (i+1, j), (i+1,j+1)
		]

		for n, m in SURROUNDING:

			if (0 <= n <= gray_img.shape[0]-1) and (0 <= m <= gray_img.shape[1]-1):
				points.append(gray_img[n,m]/255)
			else:
				points.append(gray_img[i,j]/255)

		X[count,1:] = points
		yr.append([color_img[i,j,0].tolist()])
		yg.append([color_img[i,j,1].tolist()])
		yb.append([color_img[i,j,2].tolist()])
		count +=1

# train
yr = np.array(yr)
yg = np.array(yg)
yb = np.array(yb)
alpha = 0.0001
num_iters = 20
lambda_=3
weights = dict( # initial weights
	wr=np.zeros((10,1)),
	wg=np.zeros((10,1)),
	wb=np.zeros((10,1))
)

wr = gradient_decent(X, yr, weights.get('wr'), alpha, num_iters, train='r', weights=weights, lambda_=lambda_, verbose=True)
weights['wr'] = wr # update weights
wg = gradient_decent(X, yg, weights.get('wg'), alpha, num_iters, train='g', weights=weights, lambda_=lambda_, verbose=True)
weights['wg'] = wg # update weights
wb = gradient_decent(X, yb, weights.get('wb'), alpha, num_iters, train='b', weights=weights, lambda_=lambda_, verbose=True)
weights['wb'] = wb # update weights

# test
img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
colored_img = np.zeros((img.shape[0],img.shape[1],3))
X = np.ones((1, 10), dtype=np.float64)
for i in range(img.shape[0]):
	for j in range(img.shape[1]):

		# get data from surrounding pixels
		SURROUNDING = [
			(i-1, j-1), (i-1, j), (i-1,j+1),
			(i, j-1), (i, j), (i,j+1), 
			(i+1, j-1), (i+1, j), (i+1,j+1)
		]
		pixels = []
		for n, m in SURROUNDING:

			if (0 <= n <= img.shape[0]-1) and (0 <= m <= img.shape[1]-1):
				pixels.append(img[n,m])
			else:
				pixels.append(img[i,j])
		X[0,1:] = pixels

		# predict
		r = sum(sum([wr[i]*x for i,x in enumerate(X)]))
		g = sum(sum([wg[i]*x for i,x in enumerate(X)]))
		b = sum(sum([wb[i]*x for i,x in enumerate(X)]))
		colored_img[i,j,0] = int(r)
		colored_img[i,j,1] = int(g)
		colored_img[i,j,2] = int(b)

cv2.imwrite('colorized.png',colored_img)

				
		


# SURROUNDING = [
#         (i-2, j-2), (i-2, j-1), (i-2, j), (i-2,j+1), (i-2,j+2),
#         (i-1, j-2), (i-1, j-1), (i-1, j), (i-1,j+1), (i-1,j+2), 
#         (i, j-2), (i, j-1), (i, j), (i,j+1), (i,j+2), 
#         (i+1, j-2), (i+1, j-1), (i+1, j), (i+1,j+1), (i+1,j+2),
#         (i+2, j-2), (i+2, j-1), (i+2, j), (i+2,j+1), (i+2,j+2)
#         ]