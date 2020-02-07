import cv2
import numpy as np

global points

def get_point(event,x,y,flags,param):
  global points
  if event == cv2.EVENT_LBUTTONDBLCLK:
    points.append([x,y])

def imshow(windowName, image):
	global points
	if len(points)>=2:
		return False
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, get_point)
	cv2.imshow(windowName, image)
	cv2.waitKey(3000)
	cv2.destroyAllWindows()
	return True

def part1(image):
	'''
	Part 1 - GUI for drawing line segments and obtaining 
	representation in 2D projective space

	'''
	print("GUI for obtaining 2D projective representation of a line")

	img = image.copy()

	# Projective space representation
	P2 = []
	
	global points	
	make_line = input('Press 1 for drawing line segments on image, 0 for exiting GUI :')

	while make_line:
		print("Double click on 2 points on the image and then close the window")
		
		points = []	
		show = True
		while show:
			show = imshow('image', img)
		cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0,255,0), 2)

		# Compute line parameters -> ax+by+c = 0
		x1, y1, x2, y2 = float(points[0][0]), float(points[0][1]), float(points[1][0]), float(points[1][1])
		a = y2 - y1
		b = x1 - x2
		c = (x2-x1)*y1 - (y2-y1)*x1
		P2.append([a/c, b/c, 1])

		make_line = input('Press 1 for drawing line segments on image, 0 for exiting GUI :')

	cv2.imshow('image', img)
	cv2.waitKey(0)

	P2 = np.asarray(P2)
	print("2D Projective Space representation for lines :")
	print(P2)
	
if __name__ == '__main__':

	# Read image
	img = cv2.imread('Garden.JPG')
	
	# Part 1 - GUI for drawing line segments
	part1(img)