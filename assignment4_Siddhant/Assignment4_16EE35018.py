'''
Assignment 4 : Dominant Color Transfer

Name : Siddhant Haldar - 16EE35018

'''

import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

def part1(image, k = 3):
	'''
	Function to obtain dominant colour in an image

	Input:
		image : Input image 
		k : Number of clusters for KMeans Clustering

	Output:
		labels : list of labels obtained using kmeans
		dominant_label : dominant label
		dominant pixel value in the image

	'''

	img = image.copy()

	# Convert RGB image to CIE standard
	img_cie = np.zeros((img.shape[0], img.shape[1], 2))
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			r = img[i][j][0]
			g = img[i][j][1]
			b = img[i][j][2]
			X = 0.6067*r + 0.1736*g + 0.2001*b
			Y = 0.2988*r + 0.5868*g + 0.1143*b
			Z = 0.0661 * g + 1.1149*b
			img_cie[i][j][0] = X/(X+Y+Z)
			img_cie[i][j][1] = Y/(X+Y+Z)

	# Convert image to a list of pixel values
	img_cie = img_cie.reshape((img_cie.shape[0]*img_cie.shape[1],2))

	# Use KMeans clustering to cluster and assign labels to the pixels 
	kmeans = KMeans(n_clusters = k)
	labels = kmeans.fit_predict(img_cie)

	# Calculate cluster count
	count = [0 for i in range(k)]
	for label in labels:
		count[label] += 1
	dominant_label = count.index(max(count))

	# Get list pixels representing dominant pixels
	dominant_pixel = kmeans.cluster_centers_[dominant_label]

	return labels, dominant_label, dominant_pixel

def part2(path_src, path_dst, show=True):
	'''
	Read and display source and target image.

	Input:
		path_src : Source image path
		path_tgt : Target image path

	Output:
		src : Source image
		tgt : Target image
	'''

	src = cv2.imread(path_src)
	tgt = cv2.imread(path_tgt)

	if show:
		os.makedirs('./output/', exist_ok=True)
		cv2.imwrite("./output/Part2 : Source Image.jpg", src)
		cv2.imwrite("./output/Part2 : Target Image.jpg", tgt)
		cv2.imshow("Part2 : Source Image", src)
		cv2.imshow("Part2 : Target Image", tgt)
		# cv2.waitKey(0)

	return src, tgt

def part3(image):
	'''
	Function for interactively specifying a rectangular region on the source image

	'''

	img = image.copy()

	# Select Rectangular ROI
	fromCenter = False
	r = cv2.selectROI(img, fromCenter)

	# Crop Image
	img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
	
	return img, r

def part4(image, show=True):
	'''
	Obtain pixels of the dominant colours of source image

	'''

	img = image.copy()
	h,w,_ = img.shape

	labels, dominant_label, _ = part1(img)

	# Mark dominant pixels white
	dominant_image = np.zeros((img.shape[0], img.shape[1]))
	pixels = []
	for i in range(len(labels)):
		if labels[i] == dominant_label:
			h_img, w_img = i//w, i%w
			pixels.append([h_img, w_img])
			dominant_image[h_img, w_img] = 255
	dominant_image = dominant_image.astype(np.uint8)
	
	if show:
		os.makedirs('./output/', exist_ok=True)
		cv2.imwrite("./output/Part4 : Source Cropped Region.jpg", img)
		cv2.imwrite("./output/Part4 : Source Dominant Region.jpg", dominant_image)
		cv2.imshow("Part4 : Source Cropped Region", img)
		cv2.imshow("Part4 : Source Dominant Region", dominant_image)
		# cv2.waitKey(0)

	return dominant_image, pixels

def part5(target, show=True):
	'''
	Obtain pixels of the dominant colours of source image

	'''

	tgt = target.copy()

	cropped_tgt, r_tgt = part3(tgt)
	dominant_tgt, pixels_tgt = part4(cropped_tgt, show=False)

	if show:
		os.makedirs('./output/', exist_ok=True)
		cv2.imwrite("./output/Part5 : Target Cropped Region.jpg", cropped_tgt)
		cv2.imwrite("./output/Part5 : Target Dominant Region.jpg", dominant_tgt)
		cv2.imshow("Part5 : Target Cropped Region", cropped_tgt)
		cv2.imshow("Part5 : Target Dominant Region", dominant_tgt)
		# cv2.waitKey(0)

	return cropped_tgt, dominant_tgt, pixels_tgt

def part6(source, target, pixels_src, pixels_tgt, show=True):
	'''
	Transferring the dominant color of the source region to the target region

	'''

	src = source.copy()
	tgt = target.copy()

	# Calculate mean colour
	mean = [0, 0, 0]
	for pixel in pixels_src:
		mean += src[pixel[0]][pixel[1]]
	for i in range(len(mean)):
		mean[i] = int(mean[i]/len(pixels_src))

	# Add mean colour to target image
	for pixel in pixels_tgt:
		tgt[pixel[0]][pixel[1]] = mean

	if show:	
		os.makedirs('./output/', exist_ok=True)
		cv2.imwrite("./output/Part6 : Target image changed.jpg", tgt)
		cv2.imshow("Part6 : Target image changed", tgt)
		# cv2.waitKey(0)

	return tgt

if __name__ == '__main__':
	
	path_src = "./input/IMG_6477.jpg"
	path_tgt = "./input/IMG_6481.jpg"
	show = True

	src, tgt = part2(path_src, path_tgt, show=show)
	
	cropped_src, r = part3(src)
	dominant_src, pixels_src = part4(cropped_src, show=show)
	cropped_tgt, dominant_tgt, pixels_tgt = part5(tgt, show=show)
	modified_tgt = part6(cropped_src, cropped_tgt, pixels_src, pixels_tgt, show=show)	
	cv2.waitKey(0)