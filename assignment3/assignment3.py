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

if __name__ == '__main__':
	
	img1 = cv2.imread("Amitava_first.JPG", 0)
	img2 = cv2.imread("Amitava_second.JPG", 0)
	part2(img1, img2)
