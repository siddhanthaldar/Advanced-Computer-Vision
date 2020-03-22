import numpy as np
import cv2

def part1(image1, image2):
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
	# print(keypoint2[0].pt, len(keypoint2))
	# print(len(keypoint1), len(keypoint2), len(matches))
	return keypoint1, keypoint2, good

	# for m in (good):
	# 	print(m)
	# 	print(m.queryIdx, m.trainIdx)
	# trainIdx -> keypoint 2

# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MOHR_TRIGGS/node50.html -> for part 2
def part2(image1, image2):
	kp1, kp2, matches = part1(image1.copy(), image2.copy())
	for i, match in enumerate(matches):
		x = kp1[match.queryIdx].pt[0]
		y = kp1[match.queryIdx].pt[1]
		x_dash = kp2[match.trainIdx].pt[0]
		y_dash = kp2[match.trainIdx].pt[1]

		#normalise coordinate
		# x = float((kp1[match.queryIdx].pt[0] - (image1.shape[1]/2.0))/image1.shape[1])
		# y = float((kp1[match.queryIdx].pt[1] - image1.shape[0]/2.0)/image1.shape[0])
		# x_dash = float((kp2[match.trainIdx].pt[0] - image2.shape[1]/2.0)/image2.shape[1])
		# y_dash = float((kp2[match.trainIdx].pt[1] - image2.shape[0]/2.0)/image2.shape[0])
		# print(str(x) + " " + str(y) + "    "+str(x_dash)+" "+str(y_dash))
		a = [x*x_dash, x*y_dash, x, y*x_dash, y*y_dash, y, x_dash, y_dash, 1]
		if (i == 0):
			A = np.array(a).reshape(1,-1)
		else:
			A = np.append(A, np.array(a).reshape(1,-1), axis=0)
	A_t = np.transpose(A)
	product = A_t @ A
	eigenvalue, eigenvector = np.linalg.eig(product)
	f_rough = eigenvector[-1]
	f_rough = f_rough.reshape(3, 3)
	# SVD of f_rought
	u, s, vh = np.linalg.svd(f_rough)
	# making smallest singular vector 0
	smat = np.zeros((s.shape[0], s.shape[0]), float)
	smat[0][0] = s[0]
	smat[1][1] = s[1]
	
	f = u @ smat @ vh
	return f
	# print(f.shape, eigenvector.shape, product.shape, A_t.shape, A.shape, len(matches))

def part3(image1, image2):
	kp1, kp2, matches = part1(image1.copy(), image2.copy())
	f = part2(image1, image2)
	img1 = image1.copy()
	img2 = image2.copy()
	for match in matches:
		x = kp1[match.queryIdx].pt[0]
		y = kp1[match.queryIdx].pt[1]
		x_dash = kp2[match.trainIdx].pt[0]
		y_dash = kp2[match.trainIdx].pt[1]
		point = np.array([x,y,1]).reshape(-1,1)
		point_dash = np.array([x_dash, y_dash, 1]).reshape(-1, 1)
		# https://www.cse.unr.edu/~bebis/CS791E/Notes/EpipolarGeonetry.pdf -> Page 12
		line_dash = f @ point
		line = f @ point_dash
		
		point1 = np.array([0, float(-1.0*line[2]/line[1]), 1])
		point2 = np.array([image1.shape[1]-1, float(-1.0*(line[0]*(image1.shape[1]-1) + line[2])/line[1]), 1])
		print(point1, point2)
		cv2.line(img1, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), 255, 2)

		point1_dash = [0, -1.0*line_dash[2]/line_dash[1], 1]
		point2_dash = [image2.shape[1]-1, -1.0*(line_dash[0]*(image2.shape[1]-1) + line_dash[2])/line_dash[1], 1]
		cv2.line(img2, (int(point1_dash[0]), int(point1_dash[1])), (int(point2_dash[0]),int(point2_dash[1])), 255, 2)
	cv2.imshow("Image1", img1)
	cv2.imshow("image2", img2)
	cv2.waitKey(0)



if __name__ == '__main__':
	
	# img1 = cv2.imread("Amitava_first.JPG", 0)
	# img2 = cv2.imread("Amitava_second.JPG", 0)
	img1 = cv2.imread("a.png", 0)
	img2 = cv2.imread("b.png", 0)
	print(img1.shape, img2.shape)
	part3(img1, img2)
