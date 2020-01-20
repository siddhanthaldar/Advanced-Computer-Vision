import numpy as np 
import cv2
import math

#################################### Convert image to gray scale #########################################
def part1(img):
	gray = np.zeros((img.shape[0], img.shape[1]))

	for h in range(img.shape[0]):
		for w in range(img.shape[1]):
			b = img[h][w][0]
			g = img[h][w][1]
			r = img[h][w][2]
			gray[h][w] = 0.299*r + 0.587*g + 0.114*b

	return gray.astype(np.uint8)

##################################### Scaled	bilateral	filter for	denoising #############################
class ScaledBilateralFilter:

	def __init__(self, K, sigma_s, sigma_r, sigma_G):
		self.k = K
		self.sigma_s = sigma_s
		self.sigma_r = sigma_r
		self.sigma_G = sigma_G

	def Gg(self,x1,y1,x2,y2):
		scale = 1.0 / (2 * math.pi * self.sigma_G * self.sigma_G)
		return scale * math.exp(-1.0 * ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) / (2 * self.sigma_G * self.sigma_G))

	def Gs(self,x1,y1,x2,y2):
		return math.exp(-1.0 * ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) / (2 * self.sigma_s * self.sigma_s))

	def Gr(self,I1, I2):
		return math.exp(-1.0*(I1-I2)*(I1-I2) / (2 * self.sigma_r * self.sigma_r))

	def Ig(self,img, h1, w1):
		ans = 0
		for i in range(-self.k//2, self.k//2):
			for j in range(-self.k//2, self.k//2):
				h2 = h1 + i
				w2 = w1 + j
				if h2<0 or w2<0 or h2>=img.shape[0] or w2>=img.shape[1]:
					continue
				ans += img[h2][w2] * self.Gg(h1,w1,h2,w2)
		return ans

	def Gfg(self,img, h1, w1):
		
		# Compute Numerator
		numerator = 0
		for i in range(-self.k//2, self.k//2):
			for j in range(-self.k//2, self.k//2):
				h2 = h1 + i
				w2 = w1 + j
				if h2<0 or w2<0 or h2>=img.shape[0] or w2>=img.shape[1]:
					continue
				numerator += self.Gs(h1,w1,h2,w2) * self.Gr(self.Ig(img,h2,w2), img[h1][w1]) * img[h2][w2]		

		# Compute Denominator
		denominator = 0
		for i in range(-self.k//2, self.k//2):
			for j in range(-self.k//2, self.k//2):
				h2 = h1 + i
				w2 = w1 + j
				if h2<0 or w2<0 or h2>=img.shape[0] or w2>=img.shape[1]:
					continue
				denominator += self.Gs(h1,w1,h2,w2) * self.Gr(self.Ig(img,h2,w2), img[h1][w1])

		return numerator/denominator

	def apply_filter(self,img):
		denoised_img = np.zeros(img.shape)
		for i in range(0,img.shape[0]):
			for j in range(0,img.shape[1]):
				print(i,j)
				denoised_img[i][j] = self.Gfg(img, i, j)
		return denoised_img

def part2(img):
	F = ScaledBilateralFilter(3, 4, 16, 2)
	img = F.apply_filter(img)
	return img.astype(np.uint8)


########################################### Sharpen Image ##########################################
def part3(img):
	kernel = np.array([[-1,-1,-1], 
										 [-1, 9,-1],
										 [-1,-1,-1]])

	sharpened_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			for k1 in range(-1,2):
				for k2 in range(-1,2):
					h = i + k1
					w = j + k2
					if h<0 or w<0 or h>=img.shape[0] or w>=img.shape[1]:
						continue																																																																							
					sharpened_img[i][j] += kernel[k1+1][k2+1] * img[h][w]
	return sharpened_img.astype(np.uint8)


########################################## Adaptive Thresholding #######################################
# def part4(img):
# 	binary_img = np.zeros(img.shape)
# 	for i in range(img.shape[0]):
# 		for j in range(img.shape[1]):
# 			mean = 0
# 			for k1 in range(-1,2):
# 				for k2 in range(-1,2):
# 					h = i + k1
# 					w = j + k2
# 					if h<0 or w<0 or h>=img.shape[0] or w>=img.shape[1]:
# 						continue
# 					mean += img[h][w]
# 			mean = mean/9
# 			binary_img[i][j] = 255 if img[i][j]>=mean else 0
# 	return binary_img.astype(np.uint8) 																																																																							

def otsu(img):
	'''
	Calculating threshold by maximizing between-class variance 
	(which also has the minimum within-class variance).

	'''

	# Compute histogram
	histogram = np.zeros(256)
	h,w = img.shape
	for i in range(h):
		for j in range(w):
			histogram[int(img[i][j])] += 1
	# histogram /= h*w

	# Compute threshold by maximizing between class variance
	max_variance = float('-inf')
	threshold = -1
	for i in range(0, 255):
		
		# Class 1
		weight1 = 0
		mean1 = 0
		for j in range(i+1):
			weight1 += histogram[j]
			mean1 += j * histogram[j]
		mean1 /= weight1
		weight1 /= h*w

		# Class 2
		weight2 = 0
		mean2 = 0
		for j in range(i+1, 256):
			weight2 += histogram[j]
			mean2 += j * histogram[j]
		mean2 /= weight2	
		weight2 /= h*w

		# Between class variance
		variance = weight1 * weight2 * (weight1 - weight2)**2
		if variance > max_variance:
			max_variance = variance
			threshold = i

	return threshold

def part4(img):
	# Calculate threshold using otsu algorithm
	threshold = otsu(img)
	
	# Binarize image
	binary_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] > threshold:
				binary_img[i][j] = 255
	return binary_img.astype(np.uint8)

########################################## Detection of Harris Corner points #######################################

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
		"""
		2D gaussian mask - should give the same result as MATLAB's
		fspecial('gaussian',[shape],[sigma])
		"""
		m,n = [(ss-1.)/2. for ss in shape]
		y,x = np.ogrid[-m:m+1,-n:n+1]
		h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
		h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
		sumh = h.sum()
		if sumh != 0:
				h /= sumh
		return h

def HarrisCorners(img, window_size, k, thresh):
	"""
	returns list of corners and new image with corners drawn
	window_size: The size (side length) of the sliding window
	k: Harris corner constant. Usually 0.04 - 0.06
	thresh: The threshold above which a corner is counted
	"""
	#Find x and y derivatives
	dy, dx = np.gradient(img)
	Ixx = dx**2
	Ixy = dy*dx
	Iyy = dy**2
	height = img.shape[0]
	width = img.shape[1]

	cornerList = []
	newImg = img.copy()
	color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
	offset = window_size//2

	window = matlab_style_gauss2D(shape=(5,5), sigma=0.5)

	#Loop through image and find our corners
	for y in range(offset, height-offset):
		for x in range(offset, width-offset):
			#Calculate sum of squares
			windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
			windowIxx = np.multiply(windowIxx, window)
			windowIxy = np.multiply(windowIxy, window)
			windowIyy = np.multiply(windowIyy, window)
			Sxx = windowIxx.sum()
			Sxy = windowIxy.sum()
			Syy = windowIyy.sum()

			det = (Sxx * Syy) - (Sxy**2)
			trace = Sxx + Syy
			r = det - k*(trace**2)

			if r > thresh:
				# print x, y, r
				cornerList.append([x, y, r])
				color_img.itemset((y, x, 0), 0)
				color_img.itemset((y, x, 1), 0)
				color_img.itemset((y, x, 2), 255)
	return color_img, cornerList

# def HarrisCorners(img, window_size, k, thresh):
# 	"""
# 	returns list of corners and new image with corners drawn
# 	window_size: The size (side length) of the sliding window
# 	k: Harris corner constant. Usually 0.04 - 0.06
# 	thresh: The threshold above which a corner is counted
# 	"""
# 	#Find x and y derivatives
# 	dy, dx = np.gradient(img)
# 	Ixx = dx**2
# 	Ixy = dy*dx
# 	Iyy = dy**2
# 	height = img.shape[0]
# 	width = img.shape[1]

# 	cornerList = []
# 	newImg = img.copy()
# 	color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
# 	offset = window_size//2

# 	#Loop through image and find our corners
# 	for y in range(offset, height-offset):
# 		for x in range(offset, width-offset):
# 			#Calculate sum of squares
# 			windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
# 			windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
# 			windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
# 			Sxx = windowIxx.sum()
# 			Sxy = windowIxy.sum()
# 			Syy = windowIyy.sum()

# 			det = (Sxx * Syy) - (Sxy**2)
# 			trace = Sxx + Syy
# 			r = det - k*(trace**2)

# 			if r > thresh:
# 				# print x, y, r
# 				cornerList.append([x, y, r])
# 				color_img.itemset((y, x, 0), 0)
# 				color_img.itemset((y, x, 1), 0)
# 				color_img.itemset((y, x, 2), 255)
# 	return color_img, cornerList

########################################## Connected Component ##########################################

def isokay(a, b, i, j, img, visited):
	if i>=0 and j>=0 and i<visited.shape[0] and j<visited.shape[1] and visited[i][j]==0 and (int)(img[a][b]) == (int)(img[i][j]):
		return 1
	else:
		return 0

def DFS(i, j, visited, label, connectivity):
	if connectivity == 4:
		a1 = [-1, 0, 1, 0]
		a2 = [0, 1, 0, -1]
	else:
		a1 = [-1,-1,0,1,1,1,0,-1]
		a2 = [0,-1,-1,-1,0,1,1,1]
	queue = []
	queue.append([i,j])
	while(len(queue) != 0):
		i , j = queue.pop(0)
		for l in range (len(a1)):
			if(isokay(i, j, i+a1[l], j+a2[l], img, visited)):
				# print(i, j, i+a1[l], j+a2[l])
				visited[i+a1[l]][j+a2[l]] = 1
				label[i+a1[l]][j+a2[l]] = label[i][j]
				queue.append([i+a1[l], j+a2[l]])

def part6(img, connectivity):

	visited = np.zeros(img.shape)
	label = np.zeros(img.shape)
	c = 0
	for i in range (img.shape[0]):
		for j in range (img.shape[1]):
			if(visited[i][j] == 0):
				# print(i, j)
				visited[i][j] = 1
				label[i][j] = c
				DFS(i, j, visited, label, connectivity)
				c = c+1
	return label


########################################## Erosion Dilation ##########################################

def part7a(img, kernel_size):
	'''
	Performing dilation on input image

	'''
	kernel = np.ones(kernel_size)
	new_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			flag = 1
			for l in range((-1*kernel.shape[0]//2)+1, (kernel.shape[0]//2)+1):
				for m in range((-1*kernel.shape[1]//2)+1, (kernel.shape[0]//2)+1):
					if(i+l>=0 and j+m>=0 and i+l<img.shape[0] and j+m<img.shape[1]):
						if(img[i+l][j+m] == 255):
							flag = 0
			if flag == 0:
				new_img[i][j] = 255
			else:
				new_img[i][j] = 0
	return new_img

def part7b(img, kernel_size):
	'''
	Performing erosion on input image

	'''
	kernel = np.ones(kernel_size)
	new_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			flag = 1
			for l in range((-1*kernel.shape[0]//2)+1, (kernel.shape[0]//2)+1):
				for m in range((-1*kernel.shape[1]//2)+1, (kernel.shape[0]//2)+1):
					if(i+l>=0 and j+m>=0 and i+l<img.shape[0] and j+m<img.shape[1]):
						if(img[i+l][j+m] == 0):
							flag = 0
			if flag == 0:
				new_img[i][j] = 0
			else:
				new_img[i][j] = 255
	return new_img


def part7c(img, kernel_size):
	'''
	Performing closing on input image

	'''
	closed_img = img.copy()
	closed_img = part7a(closed_img, kernel_size) # Dilation
	closed_img = part7b(closed_img, kernel_size) # Erosion
	return closed_img

def part7d(img, kernel_size):
	'''
	Performing closing on input image

	'''
	opened_img = img.copy()
	opened_img = part7b(opened_img, kernel_size) # Erosion 
	opened_img = part7a(opened_img, kernel_size)	# Dilation
	return opened_img

if __name__ == "__main__":

	# Part 1 - Read image and convert to gray Scaled
	# img = cv2.imread("images/cavepainting1.JPG")
	# gray = part1(img)
	# cv2.imwrite("Gray.jpg", gray)

	# # Part 2 - Scaled	bilateral	filter for	denoising
	# denoised_img = part2(gray)
	# cv2.imwrite("Denoised.jpg", denoised_img)
	
	# Part 3 - Sharpen image
	# img = cv2.imread("Denoised.jpg")
	# sharpened_img = part3(img)
	# cv2.imwrite("Sharpened.jpg", sharpened_img)

	# Part 4 - Adaptive Thresholding
	# img = cv2.imread("Sharpened.jpg", 0)
	# binary_img = part4(img)
	# cv2.imwrite("Binary.jpg", binary_img)

	# Part - 5 - Harris Corner
	# img = cv2.imread("checkerBoard.png", 0)
	# final_img, cornerList = HarrisCorners(img, int(5), float(0.18), int(100000))
	# cv2.imwrite("HarrisCorner.jpg", final_img)

	#Part - 6 - Connected Components 
	# img = np.array([[0,0,0,0,0,0,0,0,0,0],[0,1,1,1,0,0,1,1,1,0],[0,1,1,1,0,0,1,1,1,0],[1,0,0,1,1,0,0,0,1,0],[0,0,0,0,1,0,0,0,0,1]])#cv2.imread("input.png", 0)
	# print(img.shape)
	# print(img)
	# label = part6(img, 4)
	# print(label)

	#Part - 7a - Dilation
	# img = cv2.imread("input.png", 0)
	# print(np.unique(img))
	# kernel_size = (5,5)
	# final_img = part7a(img, kernel_size)
	# cv2.imwrite("silate.jpg", final_img)

	# #Part - 7b - Erosion
	# img = cv2.imread("input.png", 0)
	# print(np.unique(img))
	# kernel_size = (5,5)
	# final_img = part7b(img, kernel_size)
	# cv2.imwrite("erode.jpg", final_img)

	# #Part - 7c - Closing
	# img = cv2.imread("input.png", 0)
	# print(np.unique(img))
	# kernel_size = (5,5)
	# final_img = part7c(img, kernel_size)
	# cv2.imwrite("closing.jpg", final_img)

	#Part - 7d - Opening
	img = cv2.imread("input.png", 0)
	print(np.unique(img))
	kernel_size = (5,5)
	final_img = part7d(img, kernel_size)
	cv2.imwrite("opening.jpg", final_img)