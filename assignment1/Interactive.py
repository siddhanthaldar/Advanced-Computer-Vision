'''
Assignment 1 - Group 4

Script for Interactive Operation

Members:

16EE35008 - Harsh Maheshwari
16EE35018 - Siddhant Haldar

'''


import numpy as np 
import cv2
import math
import os

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
		scale = 1.0 / (2 * np.pi * self.sigma_G**2)
		return scale * np.exp(-1.0 * ((x1-x2)**2 + (y1-y2)**2) / (2 * self.sigma_G**2))

	def Gs(self,x1,y1,x2,y2):
		return np.exp(-1.0 * ((x1-x2)**2 + (y1-y2)**2) / (2 * self.sigma_s**2))

	def Gr(self,I1, I2):
		return np.exp(-1.0*(I1-I2)**2 / (2 * self.sigma_r**2))

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
		denominator = 0
		for i in range(-self.k//2, self.k//2):
			for j in range(-self.k//2, self.k//2):
				h2 = h1 + i
				w2 = w1 + j
				if h2<0 or w2<0 or h2>=img.shape[0] or w2>=img.shape[1]:
					continue
				val = self.Gs(h1,w1,h2,w2) * self.Gr(self.Ig(img,h2,w2), img[h1][w1])
				numerator += val * img[h2][w2]		
				denominator += val

		return numerator/denominator

	def apply_filter(self,img):
		denoised_img = np.zeros(img.shape)
		for i in range(0,img.shape[0]):
			for j in range(0,img.shape[1]):
				denoised_img[i][j] = self.Gfg(img, i, j)
		return denoised_img

def part2a(img, K=3, sigma_s=4, sigma_r=16, sigma_G=2):
	F = ScaledBilateralFilter(K, sigma_s, sigma_r, sigma_G)
	img = F.apply_filter(img)
	return img.astype(np.uint8)


########################################### Sharpen Image ##########################################
def part2b(img):
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


########################################### Edge Detection ##########################################

def part2c(img, threshold=50):
	kernel = np.array([[0, 1, 0], 
						[1,-4, 1],
						[0, 1, 0]])

	edge_img = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			gradient = 0
			for k1 in range(-1,2):
				for k2 in range(-1,2):
					h = i + k1
					w = j + k2
					if h<0 or w<0 or h>=img.shape[0] or w>=img.shape[1]:
						continue		
					gradient += kernel[k1+1][k2+1] * img[h][w]
			if gradient>threshold:																																																																							
				edge_img[i][j] = 255
	return edge_img.astype(np.uint8)																																																																						


########################################### Adaptive Thresholding ##########################################

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
		variance = weight1 * weight2 * (mean1 - mean2)**2
		if variance > max_variance:
			max_variance = variance
			threshold = i

	return threshold

def part2d(img):
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

def part2e(img, window_size=int(5), k=float(0.18), thresh=int(10000)):
	"""
	returns list of corners and new image with corners drawn
	window_size: The size (side length) of the sliding window
	k: Harris corner constant.
	thresh: The threshold above which a corner is counted
	"""
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
				cornerList.append([x, y, r])
				color_img.itemset((y, x, 0), 0)
				color_img.itemset((y, x, 1), 0)
				color_img.itemset((y, x, 2), 255)
	return color_img.astype(np.uint8)

########################################## Connected Component ##########################################

def isokay(a, b, i, j, img, visited):
	if i>=0 and j>=0 and i<visited.shape[0] and j<visited.shape[1] and visited[i][j]==0 and (int)(img[a][b]) == (int)(img[i][j]):
		return 1
	else:
		return 0

def DFS(i, j, img, visited, label, connectivity):
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
		for l in range(len(a1)):
			if(isokay(i, j, i+a1[l], j+a2[l], img, visited)):
				visited[i+a1[l]][j+a2[l]] = 1
				label[i+a1[l]][j+a2[l]] = label[i][j]
				queue.append([i+a1[l], j+a2[l]])

def generateLine(labels):
	lines = np.zeros(labels.shape)
	
	# Horizontal Scan
	for i in range(labels.shape[0]):
		for j in range(labels.shape[1]-1):
			if labels[i][j] != labels[i][j+1]:
				lines[i][j] = 255
					
	# Vertical Scan
	for i in range(labels.shape[0]-1):
		for j in range(labels.shape[1]):
			if labels[i][j] != labels[i+1][j]:
				lines[i][j] = 255

	return lines.astype(np.uint8)
	

def part2f(img, connectivity=4):

	visited = np.zeros(img.shape)
	labels = np.zeros(img.shape)
	c = 0
	for i in range (img.shape[0]):
		for j in range (img.shape[1]):
			if(visited[i][j] == 0):
				visited[i][j] = 1
				labels[i][j] = c
				DFS(i, j, img, visited, labels, connectivity)
				c = c+1
	return generateLine(labels) 


################################# Erosion, Dilation, Opening, Closing ##################################

def part2g_1(img, kernel_size=(3,3)):
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
	return new_img.astype(np.uint8)

def part2g_2(img, kernel_size=(3,3)):
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
	return new_img.astype(np.uint8)


def part2g_3(img, kernel_size=(3,3)):
	'''
	Performing closing on input image

	'''
	closed_img = img.copy()
	closed_img = part2g_1(closed_img, kernel_size) # Dilation
	closed_img = part2g_2(closed_img, kernel_size) # Erosion
	return closed_img.astype(np.uint8)

def part2g_4(img, kernel_size=(3,3)):
	'''
	Performing closing on input image

	'''
	opened_img = img.copy()
	opened_img = part2g_2(opened_img, kernel_size) # Erosion 
	opened_img = part2g_1(opened_img, kernel_size)	# Dilation
	return opened_img.astype(np.uint8)

if __name__ == "__main__":
	os.makedirs("interactiveOutput", exist_ok = True)
	vis = int(input("If you want to visualise output PRESS 1 else 0: "))

	# Part 1 - Read image and convert to gray Scaled
	print("Converting image to Grayscale")
	img = cv2.imread("images/cavepainting1.JPG")
	if vis:
		cv2.imshow("Input", img)	
	img = part1(img)
	if vis:
		print("To close window press any key")
		cv2.imshow("Gray scale", img)
		cv2.waitKey()
		cv2.destroyAllWindows()
	cv2.imwrite("interactiveOutput/Gray.jpg", img)

	# Interactive UI
	print("For further operations, use the following codes :\n\n\
				 0 : Exit\n\
				 1  : Scaled Bilateral Filtering\n\
				 2  : Image Sharpening\n\
				 3  : Edge Extraction\n\
				 4  : Binarization by Adaptive Thresholding\n\
				 5  : Detection of Harris Corner points\n\
				 6  : Connected Component Compututation\n\
				 7  : Dilation\n\
				 8  : Erosion\n\
				 9  : Closing\n\
				 10 : Opening\n\
		")

	func = {
		1 : part2a,
		2 : part2b,
		3 : part2c,
		4 : part2d,
		5 : part2e,
		6 : part2f,
		7 : part2g_1,
		8 : part2g_2,
		9 : part2g_3,
		10 : part2g_4
	}

	operation_name = {
		0 : "Exit",
		1 : "Scaled Bilateral Filtering",
		2 : "Image Sharpening",
		3 : "Edge Extraction",
		4 : "Binarization using Adaptive Threshold",
		5 : "Detection of Harris Corner Points",
		6 : "Connected Components",
		7 : "Dilation",
		8 : "Erosion",
		9 : "Closing",
		10 : "Opening"
	}

	while(1):
		fvalue = int(input("Operation Number: "))
		if fvalue == 0:
			break

		f = func[fvalue]
		print("Performing : ", operation_name[fvalue])
		img = f(img)
		if vis:
			cv2.imshow(operation_name[fvalue]+".jpg", img)	
			cv2.waitKey()
			cv2.destroyAllWindows()
		cv2.imwrite("interactiveOutput/"+operation_name[fvalue]+".jpg", img)

	print("Exiting Code")
	
