'''
Group 4 : Assignment 3

Members : (1) Harsh Maheshwari - 16EE35008
					(2) Siddhant Haldar - 16EE35018
'''

import numpy as np
import cv2
import math
import os

def part1(image1, image2, show=True):
	if show:
		print("*********************************************************************")
		print("Part 1 : Key	point	extraction and	feature	descriptors")
	
	img1 = image1.copy()
	img2 = image2.copy()
	sift = cv2.xfeatures2d.SIFT_create()
	keypoint1, desc1 = sift.detectAndCompute(img1,None)
	keypoint2, desc2 = sift.detectAndCompute(img2,None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(desc1,desc2, k=2)
	
	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	if show:
		print("Done")
		print("*********************************************************************")

	return keypoint1, keypoint2, good

# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MOHR_TRIGGS/node50.html -> for part 2
def part2(image1, image2, show=True):
	if show:
		print("*********************************************************************")
		print("Part 2 : Fundamental Matrix Computation")
	
	kp1, kp2, matches = part1(image1.copy(), image2.copy(), show=False)

	# Calculate sigma_x and sigma_y for both images
	x = []
	y = []
	x_dash = []
	y_dash = []
	for i, match in enumerate(matches):
		x.append(kp1[match.queryIdx].pt[0])
		y.append(kp1[match.queryIdx].pt[1])
		x_dash.append(kp2[match.trainIdx].pt[0])
		y_dash.append(kp2[match.trainIdx].pt[1])

	# Different normalization scheme
	sigma_x = np.std(x)
	sigma_y = np.std(y)
	mean_x = np.mean(x)
	mean_y = np.mean(y)
	T1 = np.array([[math.sqrt(2.0)/sigma_x, 0 , -math.sqrt(2.0)*mean_x/sigma_x],
                 [0, math.sqrt(2.0)/sigma_y , -math.sqrt(2.0)*mean_y/sigma_y],
                 [0 ,  0 , 1]])
	
	sigma_x = np.std(x_dash)
	sigma_y = np.std(y_dash)
	mean_x = np.mean(x_dash)
	mean_y = np.mean(y_dash)
	T2 = np.array([[math.sqrt(2.0)/sigma_x, 0 , -math.sqrt(2.0)*mean_x/sigma_x],
                 [0, math.sqrt(2.0)/sigma_y , -math.sqrt(2.0)*mean_y/sigma_y],
                 [0 ,  0 , 1]])
	
	# # Normalization matrices for range [-1,1]
	# T1 = np.array([[2.0/image1.shape[1], 0 , -1],
	# 							 [0, 2.0/image1.shape[0] , -1],
	# 							 [0 ,  0 , 1]])
	# T2 = np.array([[2.0/image2.shape[1], 0 , -1],
	# 							 [0, 2.0/image2.shape[0] , -1],
	# 							 [0 ,  0 , 1]])

	for i, match in enumerate(matches):
		# the order of keypoint is (x,y)
		x = kp1[match.queryIdx].pt[0]
		y = kp1[match.queryIdx].pt[1]
		x_dash = kp2[match.trainIdx].pt[0]
		y_dash = kp2[match.trainIdx].pt[1]

		point = np.array([x,y,1]).reshape(-1,1)
		point_dash = np.array([x_dash, y_dash, 1]).reshape(-1,1)

		# Normalize points
		point = T1 @ point
		point_dash = T2 @ point_dash
		x = point[0][0]/point[2][0]
		y = point[1][0]/point[2][0]
		x_dash = point_dash[0][0]/point_dash[2][0]
		y_dash = point_dash[1][0]/point_dash[2][0]
		
		# Create A matrix
		a = [x*x_dash, x*y_dash, x, y*x_dash, y*y_dash, y, x_dash, y_dash, 1]
		if (i == 0):
			A = np.array(a).reshape(1,-1)
		else:
			A = np.append(A, np.array(a).reshape(1,-1), axis=0)

	# Solve to obtain rough value of Fundamental Matrix
	A_t = np.transpose(A)
	product = A_t @ A
	eigenvalue, eigenvector = np.linalg.eig(product)
	f_rough = eigenvector[-1]
	f_rough = f_rough.reshape(3, 3)

	# Denormalize Fundamental Matrix
	f_rough = T2.T @ f_rough @ T1

	# SVD of f_rough
	u, s, vh = np.linalg.svd(f_rough)
	# making smallest singular vector 0 to obtain a rank 2 matrix
	smat = np.zeros((u.shape[1], vh.shape[0]), float)
	smat[0][0] = s[0]
	smat[1][1] = s[1]
	
	f = u @ smat @ vh

	if show:
		print("Fundamental Matrix :")
		print(f)
		print("Done")
		print("*********************************************************************")

	return f

def part3(image1, image2, show=True):
	if show:
		print("*********************************************************************")
		print("Part 3 : Drawing Epipolar Lines")
	
	kp1, kp2, matches = part1(image1.copy(), image2.copy(), show=False)
	f = part2(image1.copy(), image2.copy(), show=False)
	img1 = image1.copy()
	img2 = image2.copy()
	line_list = list()
	line_dash_list = list()
	for match in matches:
		x = kp1[match.queryIdx].pt[0]
		y = kp1[match.queryIdx].pt[1]
		x_dash = kp2[match.trainIdx].pt[0]
		y_dash = kp2[match.trainIdx].pt[1]

		point = np.array([x,y,1]).reshape(-1,1)
		point_dash = np.array([x_dash,y_dash,1]).reshape(-1, 1)

		# Obtain epipolar lines
		# https://www.cse.unr.edu/~bebis/CS791E/Notes/EpipolarGeonetry.pdf -> Page 12
		line_dash = f @ point
		line = np.transpose(f) @ point_dash
		line_list.append(line)
		line_dash_list.append(line_dash)

		point1 = np.array([0, float(-1.0*line[2]/line[1]), 1])
		point2 = np.array([image1.shape[1]-1, float(-1.0*(line[0]*(image1.shape[1]-1) + line[2])/line[1]), 1])
		cv2.line(img1, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), 255, 2)

		point1_dash = [0, -1.0*line_dash[2]/line_dash[1], 1]
		point2_dash = [image2.shape[1]-1, -1.0*(line_dash[0]*(image2.shape[1]-1) + line_dash[2])/line_dash[1], 1]
		cv2.line(img2, (int(point1_dash[0]), int(point1_dash[1])), (int(point2_dash[0]),int(point2_dash[1])), 255, 2)

	if show:
		os.makedirs('./output', exist_ok=True)
		cv2.imwrite('./output/Part3-EpipolarLines_Image1.jpg', img1)
		cv2.imwrite('./output/Part3-EpipolarLines_Image2.jpg', img2)
		cv2.imshow("Part 3 : Epipolar Lines in Image 1", img1)
		cv2.imshow("Part 3 : Epipolar Lines in Image 2", img2)
		# cv2.waitKey(0)
		print("Done")
		print("*********************************************************************")
		
	return line_list, line_dash_list	

def part4(image1, image2, show=True):
	if show:
		print("*********************************************************************")
		print("Part 4 : Error in epipole estimation from Fundamental Matrix and from epipoles")
		
	f = part2(image1.copy(), image2.copy(), show=False)
	lines, lines_dash = part3(image1.copy(), image2.copy(), show=False)
	img1 = image1.copy()
	img2 = image2.copy()
	
	# Calculate epipoles from epipolar lines
	e_line = np.cross(lines[0].reshape(-1), lines[1].reshape(-1))
	e_line /= e_line[2]
	e_dash_line = np.cross(lines_dash[0].reshape(-1), lines_dash[1].reshape(-1))
	e_dash_line /= e_dash_line[2]
	
	'''
	Calculate epipoles from Fundamental Matrix
	Links :
		https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook1/HZepipolar.pdf -> page 8
		e_dash.T -> left null space of fundamental matrix and e is right null space of fundamental matrix
		https://cseweb.ucsd.edu/classes/wi15/cse252B-a/nullspace.pdf -> to find right and left null space of matrix'
	'''
	u, s, vh = np.linalg.svd(f)
	#only one 0 singular value must be there, taking the minimum one
	index = -1
	# ith column
	e_dash_transpose_f = u[:, index]
	#ith row
	e_f = vh[index, :]
	e_f /= e_f[2]
	e_dash_f = np.transpose(e_dash_transpose_f)
	e_dash_f /= e_dash_f[2]

	# Distances between different estimated values of epipoles
	distance_e = ((e_line[0]-e_f[0])**2 + (e_line[1]-e_f[1])**2)**0.5
	distance_e_dash = ((e_dash_line[0]-e_dash_f[0])**2 + (e_dash_line[1]-e_dash_f[1])**2)
	
	if show:
		print("Epipoles from Lines : ", [int(e_line[0]), int(e_line[1])], [int(e_dash_line[0]), int(e_dash_line[1])])
		print("Epipoles from Fundamental Matrix : ", [int(e_f[0]), int(e_f[1])], [int(e_dash_f[0]), int(e_dash_f[1])])
		print("Distance between estimated values of e :",distance_e)
		print("Distance between estimated values of e_dash :", distance_e_dash)
		print("Done")
		print("*********************************************************************")
		
	# Returning epipoles obtained from Fundamental Matrix for use in part 5
	return e_f, e_dash_f

def part5(image1, image2, show=True):
	if show:
		print("*********************************************************************")
		print("Part 5 : Estimating projection matrices from Fundamental	matrix")
	
	F = part2(image1.copy(), image2.copy(), show=False)
	e, e_dash = part4(image1.copy(), image2.copy(), show=False)

	P = np.array([[1,0,0,0],
		            [0,1,0,0],
		            [0,0,1,0]], dtype= np.float)

	S = np.array([[0,-e_dash[2],e_dash[1]],
		            [e_dash[2],0,-e_dash[0]],
		            [-e_dash[1],e_dash[0],0]])

	e_dash = e_dash.reshape(-1,1)

	P_dash = np.append(S@F, e_dash, axis=1)

	if show:
		print("P :", P)
		print("P_dash:", P_dash)
		print("Done")
		print("*********************************************************************")
	return P, P_dash

if __name__ == '__main__':
	
	img1 = cv2.imread("./input/Amitava_first.JPG", 0)
	img2 = cv2.imread("./input/Amitava_second.JPG", 0)

	# Part 1 : Key	point	extraction and	feature	descriptors
	kp1, kp2, matches = part1(img2, img2)

	# Part 2 : Fundamental Matrix Computation
	F = part2(img1, img2)

	# Part 3 : Drawing Epipolar Lines
	line_list, line_dash_list = part3(img1, img2)

	# Part 4 : Computing epipoles and the distance between them
	e, e_dash = part4(img1, img2)

	# Part 5 : Estimating projection matrices from Fundamental	matrix
	P, P_dash = part5(img1, img2)

	cv2.waitKey(0)
	