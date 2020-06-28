'''
Assignment 5 : Range Image Processing

Name : Siddhant Haldar - 16EE35018

'''

import numpy as np
import cv2
import os

##################################### Part 1 : Curvature Computation ################################

def part1(path_range):
	'''
	 Compute principal, Gaussian and mean curvatures at each point of the range data
	
	'''
	
	range_img = cv2.imread(path_range,0)

	# Sobel filters for calculating derivatives
	sobel_v = np.array([[1,0,-1],
											[2,0,-2],
											[1,0,-1]])

	sobel_u = np.array([[1,2,1],
											[0,0,0],
											[-1,-2,-1]])

	# Pad image with zeros before applying filter
	padded_image = range_img.copy()
	padded_image = np.pad(padded_image,((1,1),(1,1)), 'constant', constant_values=((0,0),(0,0))).astype(np.uint8)

	# Derivative images
	Iu = np.zeros(range_img.shape)
	Iv = np.zeros(range_img.shape)
	Iuu = np.zeros(range_img.shape)
	Iuv = np.zeros(range_img.shape)
	Ivv = np.zeros(range_img.shape)

	# Compute Ix and Iy
	for i in range(1, padded_image.shape[0]-1):
		for j in range(1, padded_image.shape[1]-1):
			mat = padded_image[i-1:i+2, j-1:j+2]
			Iu[i-1][j-1] = np.sum(np.multiply(mat, sobel_u))
			Iv[i-1][j-1] = np.sum(np.multiply(mat, sobel_v))

	# Compute Ixx, Ixy and Iyy
	Iu_padded = Iu.copy()
	Iu_padded = np.pad(Iu_padded,((1,1),(1,1)), 'constant', constant_values=((0,0),(0,0))).astype(np.uint8)
	Iv_padded = Iv.copy()
	Iv_padded = np.pad(Iv_padded,((1,1),(1,1)), 'constant', constant_values=((0,0),(0,0))).astype(np.uint8)
	for i in range(1, Iu_padded.shape[0]-1):
		for j in range(1, Iv_padded.shape[1]-1):
			mat = Iu_padded[i-1:i+2, j-1:j+2]
			Iuu[i-1][j-1] = np.sum(np.multiply(mat, sobel_u))
			Iuv[i-1][j-1] = np.sum(np.multiply(mat, sobel_v))
			mat = Iv_padded[i-1:i+2, j-1:j+2]
			Ivv[i-1][j-1] = np.sum(np.multiply(mat, sobel_v))

	############# Calculation principal, gaussian and mean curvatures and characterize topology ######################
	p1_curvature = np.zeros(range_img.shape)
	p2_curvature = np.zeros(range_img.shape)
	g_curvature = np.zeros(range_img.shape)
	m_curvature = np.zeros(range_img.shape)
	mg_topology = np.zeros(range_img.shape)  # topology based on mean and gaussian curvatures
	p_topology = np.zeros(range_img.shape)   # topology based on principal curvatures

	for i in range(range_img.shape[0]):
		for j in range(range_img.shape[1]):
			hu = Iu[i][j]
			hv = Iv[i][j]
			huu = Iuu[i][j]
			huv = Iuv[i][j]
			hvv = Ivv[i][j]			

			# Obtain Linear Map
			den = np.sqrt(1+hu**2+hv**2)
			e = -1.0*huu/den
			f = -1.0*huv/den
			g = -1.0*hvv/den
			E = 1 + hu**2
			F = hu * hv
			G = 1 + hv**2
			linear_map = np.array([[e,f],[f,g]])@np.linalg.inv(np.array([[E,F],[F,G]]))

			# Compute K and H from linear map
			K = np.linalg.det(linear_map)
			H = 0.5 * np.trace(linear_map)

			# Compute pixel wise curvatures
			factor = (H**2-K) if (H**2-K)>=0 else 0
			factor = np.sqrt(factor)
			g_curvature[i][j] = K
			m_curvature[i][j] = H
			p1_curvature[i][j] = H + factor
			p2_curvature[i][j] = H - factor

			# Characterise topology based on mean and gaussian curvatures
			if H<0 and K<0: # Peak Surface
				mg_topology[i][j] = 1
			elif H<0 and K==0: # Ridge Surface
				mg_topology[i][j] = 2
			elif H<0 and K>0: # Saddle Ridge
				mg_topology[i][j] = 3
			elif H==0 and K==0: # Flat Surface
				mg_topology[i][j] = 4
			elif H==0 and K>0: # Minimal Surface
				mg_topology[i][j] = 5
			elif H>0 and K<0: # Pit Surface
				mg_topology[i][j] = 6
			elif H>0 and K==0: # Valley Surface
				mg_topology[i][j] = 7
			elif H>0 and K>0: # Saddle Valley
				mg_topology[i][j] = 8

			# Characterise topology based on principal curvatures
			if p1_curvature[i][j]<0 and p2_curvature[i][j]<0: # Peak 
				p_topology[i][j] = 1
			elif p1_curvature[i][j]<0 and p2_curvature[i][j]==0: # Ridge 
				p_topology[i][j] = 2
			elif p1_curvature[i][j]<0 and p2_curvature[i][j]>0: # Saddle 
				p_topology[i][j] = 3
			elif p1_curvature[i][j]<0 and p2_curvature[i][j]==0: # Ridge
				p_topology[i][j] = 4
			elif p1_curvature[i][j]==0 and p2_curvature[i][j]==0: # Flat
				p_topology[i][j] = 5
			elif p1_curvature[i][j]==0 and p2_curvature[i][j]>0: # Valley
				p_topology[i][j] = 6
			elif p1_curvature[i][j]>0 and p2_curvature[i][j]<0: # Saddle
				p_topology[i][j] = 7
			elif p1_curvature[i][j]>0 and p2_curvature[i][j]==0: # Valley
				p_topology[i][j] = 8
			elif p1_curvature[i][j]>0 and p2_curvature[i][j]>0: # Pit
				p_topology[i][j] = 9

	#############################################################################

	return range_img, [p1_curvature, p2_curvature, g_curvature, m_curvature], [mg_topology, p_topology]


########################## Part 2 : Neighbourhood Plane Set (NPS) Computation ################################

def part2(range_image, k=6):
	'''
	 Compute NPS and binarized image corresponding to NPS
	
	'''

	range_img = range_image.copy()
	
	# DNP Definition. 
	# For each type, a list of 8 pixel changes [dx, dy. dz] are stored
	DNP = [[[0,0,-1], [1,0,-1], [1,0,0], [1,0,1], [0,0,1], [-1,0,1], [-1,0,0], [-1,0,-1]],
				 [[0,0,-1], [0,1,-1], [0,1,0], [0,1,1], [0,0,1], [0,-1,1], [0,-1,0], [0,-1,-1]],
				 [[-1,0,0], [-1,1,0], [0,1,0], [1,1,0], [1,0,0], [1,-1,0], [0,-1,0], [-1,-1,0]],
				 [[-1,-1,0], [-1,-1,-1], [0,0,-1], [1,1,-1], [1,1,0], [1,1,1], [0,0,1], [-1,-1,1]],
				 [[1,-1,0], [1,-1,-1], [0,0,-1], [-1,1,-1], [-1,1,0], [-1,1,1], [0,0,1], [1,-1,1]],
				 [[-1,0,-1], [-1,1,-1], [0,1,0], [1,1,1], [1,0,1], [1,-1,1], [0,-1,0], [-1,-1,-1]],
				 [[1,0,-1], [1,1,-1], [0,1,0], [-1,1,1], [-1,0,1], [-1,-1,1], [0,-1,0], [1,-1,-1]],
				 [[-1,0,0], [-1,1,-1], [0,1,-1], [1,1,-1], [1,0,0], [1,-1,1], [0,-1,1], [-1,-1,1]],
				 [[-1,0,0], [-1,1,1], [0,1,1], [1,1,1], [1,0,0], [1,-1,-1], [0,-1,-1], [-1,-1,-1]],
				]
	
	NPS_image = np.zeros(range_img.shape)
	NPS = []
	for i in range(range_img.shape[0]):
		NPS.append([])
		for j in range(range_img.shape[1]):
			NPS[i].append([])

			# Compute points available in neighbourhood
			neighbourhood = np.zeros((3,3))
			for h in [-1,0,1]:
				for w in [-1,0,1]:
					if h==0 and w==0:
						neighbourhood[1][1] = 1
					elif i+h>=0 and i+h<range_img.shape[0]:
						if j+w>=0 and j+w<range_img.shape[1]:
							diff = int(range_img[i+h][j+w]) - int(range_img[i][j])
							if abs(diff)<=1:
								neighbourhood[h+1][w+1] = diff

			# Append planes to NPS which exist in neighbourhood
			for idx, case in enumerate(DNP):
				count = 0
				for instance in case:
					if neighbourhood[instance[0]+1][instance[1]+1] == instance[2]:
						count+=1
						if count>=k:
							break
				if count>=k:
					NPS[i][j].append(idx)

			# Add number element corresponding to NPS in NPS image
			for plane in NPS[i][j]:
				NPS_image[i][j] += 2**plane

	return NPS, NPS_image		


############################# Part 3 : Segmentation of Range Images ##############################

def isokay(a, b, i, j, range_img, feature_image, visited, gap):
	if i>=0 and j>=0 and i<visited.shape[0] and j<visited.shape[1] and visited[i][j]==0 and abs(feature_image[a][b] - feature_image[i][j])<gap and abs(int(range_img[a][b])-int(range_img[i][j]))<=1:
	# if i>=0 and j>=0 and i<visited.shape[0] and j<visited.shape[1] and visited[i][j]==0 and feature_image[a][b] == feature_image[i][j] and abs(int(range_img[a][b]) - int(range_img[i][j]))<=1:
		return 1
	else:
		return 0

def DFS(i, j, range_img, feature_image, visited, label, gap):
	a1 = [-1,-1,0,1,1,1,0,-1]
	a2 = [0,-1,-1,-1,0,1,1,1]
	queue = []
	queue.append([i,j])
	while(len(queue) != 0):
		i , j = queue.pop(0)
		for l in range(len(a1)):
			if(isokay(i, j, i+a1[l], j+a2[l], range_img, feature_image, visited, gap)):
				# print(i, j, i+a1[l], j+a2[l])
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

def part3(range_img, feature_image):
	'''
	Region growing of homogenous labels using specific features of the image

	'''

	img = range_img.copy()

	gap = (np.max(feature_image)-np.min(feature_image))/10.0

	visited = np.zeros(img.shape)
	labels = np.zeros(img.shape)
	c = 0
	for i in range (img.shape[0]):
		for j in range (img.shape[1]):
			if(visited[i][j] == 0):
				# print(i, j)
				visited[i][j] = 1
				labels[i][j] = c
				DFS(i, j, img, feature_image, visited, labels, gap)
				c = c+1
	return labels, generateLine(labels) 


##################################### Main Function #################################

if __name__ == '__main__':

	images = [str(i) for i in range(5)]

	for img_num in images:
		print("Image " + img_num)
	
		path_image = "./RGBD_dataset/" + img_num+ ".jpg"
		path_range = "./RGBD_dataset/" + img_num+ ".png"

		# Part 1 :  Compute principal, Gaussian and mean curvatures
		range_image, [p1_curvature, p2_curvature, g_curvature, m_curvature], [mg_topology, p_topology] = part1(path_range)

		# Part 2 : Compute Neighbourhood Plane Set (NPS) at each pixel 
		NPS, NPS_image = part2(range_image)
		
		# Part 3a :  Perform region growing of homogeneous labels using principal curvatures 
		labels, lines_p1 = part3(range_image, p1_curvature)
		labels, lines_p2 = part3(range_image, p2_curvature)
		
		# Part 3b :  Perform region growing of homogeneous labels using Gaussian curvatures 
		labels, lines_g = part3(range_image, g_curvature)
		
		# Part 3c :  Perform region growing of homogeneous labels using NPS 
		labels, lines_nps = part3(range_image, NPS_image)

		# Part 4 : Case considering both principal and Gaussian curvatures
		labels, lines_both = part3(range_image, p1_curvature + g_curvature)		

		# Save Images
		os.makedirs('./output/', exist_ok=True)
		cv2.imwrite('./output/' + img_num + '_Range.jpg', range_image)
		cv2.imwrite('./output/' + img_num + '_Principal1.jpg', lines_p1)
		cv2.imwrite('./output/' + img_num + '_Principal2.jpg', lines_p2)
		cv2.imwrite('./output/' + img_num + '_Gaussian.jpg', lines_g)
		cv2.imwrite('./output/' + img_num + '_NPS.jpg', lines_nps)
		cv2.imwrite('./output/' + img_num + '_PrincipalGaussian.jpg', lines_both)
