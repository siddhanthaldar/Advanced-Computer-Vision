import numpy as np
import cv2



if __name__ == '__main__':
	
	img1 = cv2.imread("Amitava_first.JPG", 0)
	img2 = cv2.imread("Amitava_second.JPG", 0)
	part2(img1, img2)
