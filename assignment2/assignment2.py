import cv2
import numpy as np

#################### Part 1 : GUI for obtaning 2D Projective Space Representation ################

global points

def get_point(event,x,y,flags,param):
  global points
  if event == cv2.EVENT_LBUTTONDBLCLK:
    points.append([x,y])

def imshow(windowName, image, numLines):
	global points
	if len(points)>=2*numLines:
		return False
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, get_point)
	cv2.imshow(windowName, image)
	cv2.waitKey(1000 * 2 * numLines)
	cv2.destroyAllWindows()
	return True


def part1(image, numLines):
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
			show = imshow('image', img, numLines)
		cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0,255,0), 2)

		for i in range(len(points)/2):
			x1, y1, x2, y2 = float(points[2*i][0]), float(points[2*i][1]), float(points[2*i+1][0]), float(points[2*i+1][1])
			cv2.line(img, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 2)
			
			# Compute line parameters -> ax+by+c = 0
			# a = y2 - y1
			# b = x1 - x2
			# c = (x2-x1)*y1 - (y2-y1)*x1
			p = np.cross([x1,y1,1], [x2,y2,1])
			p /= p[2]
			P2.append(p)

		make_line = input('Press 1 for drawing line segments on image, 0 for exiting GUI :')

	cv2.imshow('image', img)
	cv2.waitKey(0)

	P2 = np.asarray(P2)
	
	return P2

############################ Part 2 : Vanishing Line ##############################################

def part2(image):
	'''
	Obtain vanishing line for an image
	
	'''
	img = image.copy()
	
	P2 = part1(img, 4)

	point1 = np.cross(P2[0], P2[1])
	point1 /= point1[2]
	point2 = np.cross(P2[2], P2[3])
	point2 /= point2[2]
	
	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)

	cv2.imshow('vanishing line', img)
	cv2.waitKey(0)
	
	line = np.cross([point1[0], point1[1], 1], [point2[0], point2[1], 1])
	
	return line


############################ Part 3 : Line Parallel to Vanishing Line ##################################

def part3(image):

	img = image.copy()

	line = part2(img)

	center = [img.shape[1]//2, img.shape[0]//2]
	line[2] = -1.0 * (line[0] * center[0] + line[1] * center[1])

	point1 = [0, -1.0*line[2]/line[1]]
	point2 = [img.shape[1]-1, -1.0*(line[0]*(img.shape[1]-1) + line[2])/line[1]]

	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)

	cv2.imshow('parallel line', img)
	cv2.waitKey(0)
	
	
if __name__ == '__main__':

	# Read image
	img = cv2.imread('Garden.JPG')
	
	# Part 1 : GUI for drawing line segments
	# P2 = part1(img, 2)

	# Part 2 : Vanishing Line
	# line = part2(img)

	# Part 3 : Parallel line to vanishing line
	part3(img)