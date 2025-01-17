import os
import cv2
import random
import numpy as np

#################### Part 1 : GUI for obtaning 2D Projective Space Representation ################

global points
global vanish_line
global centre_line
global save_point

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
	cv2.waitKey(0)
	# cv2.waitKey(1000 * 2 * numLines)
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
	make_line = int(input('Press 2 for developers mode, 1 for drawing line segments on image, 0 for exiting GUI :'))

	if make_line == 2:
		p1 = [228.0, 407.0]
		p2 = [778.0, 422.0]
		p3 = [45.0, 600.0]
		p4 = [898.0, 601.0]
		p5 = [787.0, 417.0]
		p6 = [896.0, 592.0]
		p7 = [222.0, 415.0]
		p8 = [37.0, 577.0]

		points = [[p1,p2], [p3, p4], [p5, p6], [p7,p8]]

		for i in range(len(points)):
			x1, y1, x2, y2 = float(points[i][0][0]), float(points[i][0][1]), float(points[i][1][0]), float(points[i][1][1])
			cv2.line(img, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 2)
			
			# Compute line parameters -> ax+by+c = 0
			p = np.cross([x1,y1,1], [x2,y2,1])
			p /= p[2]
			P2.append(p)    

	else:

		while make_line:
			print("Double click on " +  str(numLines*2) + " points on the image and then close the window")
			
			points = [] 
			imshow('image', img, numLines)
			
			while len(points) < 2*numLines:
				print("Only " + str(len(points)) + " points have been selected. Reselect " + str(numLines*2) + " points on the image and then close the window")
				points = [] 
				show = imshow('image', img, numLines)
					
			for i in range(int(len(points)/2)):
				x1, y1, x2, y2 = float(points[2*i][0]), float(points[2*i][1]), float(points[2*i+1][0]), float(points[2*i+1][1])
				cv2.line(img, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 2)
				
				# Compute line parameters -> ax+by+c = 0
				p = np.cross([x1,y1,1], [x2,y2,1])
				p /= p[2]
				P2.append(p)

			# make_line = int(input('Press 1 for drawing line segments on image, 0 for exiting GUI :'))

	cv2.imshow('image_part1', img)
	cv2.imwrite('Images/LinesConsideredForVanishingLine.jpg', img)
	cv2.waitKey(0)

	P2 = np.asarray(P2)
	
	return P2

############################ Part 2 : Vanishing Line ##############################################

def part2(image):
	'''
	Obtain vanishing line for an image
	
	'''
	global vanish_line
	img = image.copy()
	
	P2 = part1(img, 4)

	point1 = np.cross(P2[0], P2[1])
	point2 = np.cross(P2[2], P2[3])
	line = np.cross([point1[0], point1[1], 1], [point2[0], point2[1], 1])
	vanish_line = line.copy()
	point1 /= point1[2]
	point2 /= point2[2]
	
	point1 = [0, -1.0*line[2]/line[1], 1]
	point2 = [img.shape[1]-1, -1.0*(line[0]*(img.shape[1]-1) + line[2])/line[1], 1]

	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)

	cv2.imshow('vanishing line', img)
	cv2.imwrite('Images/VanishingLine.jpg', img)
	cv2.waitKey(0)
	
	return line


############################ Part 3 : Line Parallel to Vanishing Line ##################################

def part3(image, line):
	global centre_line
	img = image.copy()

	# line = part2(img)

	center = [img.shape[1]//2, img.shape[0]//2]
	line[2] = -1.0 * (line[0] * center[0] + line[1] * center[1])

	point1 = [0, -1.0*line[2]/line[1], 1]
	point2 = [img.shape[1]-1, -1.0*(line[0]*(img.shape[1]-1) + line[2])/line[1], 1]

	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)

	centre_line = np.cross(point1, point2)
	centre_line /= centre_line[2]

	cv2.imshow('parallel line', img)
	cv2.imwrite('Images/VanishingLineThroughCentre.jpg', img)
	cv2.waitKey(0)


############################ Part 4 : Obtain lines corresponding to a Vanishing Point ##################################

def get_point1(event,x,y,flags,param):
	global save_point
	if event == cv2.EVENT_LBUTTONDBLCLK:
		save_point.append([y, x])


def take_point(windowName, img):
	global centre_line
	global save_point

	point1 = [0, -1.0*centre_line[2]/centre_line[1], 1]
	point2 = [img.shape[1]-1, -1.0*(centre_line[0]*(img.shape[1]-1) + centre_line[2])/centre_line[1], 1]
	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)

	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, get_point1)
	cv2.imshow(windowName, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return True

def part4(image):
	global centre_line
	global vanish_line
	global save_point

	img = image.copy()
	var_image = image.copy()
	print("Choose a point on the line drawn in image and close the window(Press any key).")

	save_point = []
	take_point("Pick_point", var_image)
	if len(save_point)<1:
		print("Point NOT chosen. Please choose a point on the line drawn in image and close the window(Press any key).")
		save_point = []
		take_point("Pick point", var_image)

	variable = save_point[0][1]
	cv2.line(img, (int(variable), int(0)), (int(variable), int(img.shape[0] - 1)), (51,255,255), 2)
	p = [variable, -1.0*(vanish_line[0]*(variable) + vanish_line[2])/vanish_line[1], 1]

	# Mark Vanishing Line on Image
	point1 = [0, -1.0*vanish_line[2]/vanish_line[1], 1]
	point2 = [img.shape[1]-1, -1.0*(vanish_line[0]*(img.shape[1]-1) + vanish_line[2])/vanish_line[1], 1]
	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)
	

	for i in range(3):
		H = np.random.rand(2,3)*10*(i+1)
		# H = np.array([[1,0,0],[0,1,0]])*100
		a = np.array([-p[1], p[0]-(1.0/p[1]) ,1]).astype(np.float32).reshape(1,-1)
		H = np.append(H, a, axis = 0)
		result_point = np.matmul(H, np.transpose(p))
		projected_a = -result_point[1]
		projected_b = result_point[0]
		# print("H-: ", H)
		# print("Result Point -: ", result_point)
		# print("projected_b -: ", projected_b)
		# print("projected_a -: ", projected_a)
		B = int(random.randint(0,50)*random.randint(0,5))
		G = int(random.randint(0,50)*random.randint(0,5))
		R = int(random.randint(0,50)*random.randint(0,5))


		for c in range(2):
			l = np.matmul(np.transpose(H), np.array([projected_a, projected_b, c*(projected_a+projected_b)]).reshape(-1, 1))
			a = l[0]
			b = l[1]
			
			point1 = (-l[2]-(img.shape[0]-1)*l[1])/a

			if point1 >= img.shape[1]:
				point1 = (-l[2]-(img.shape[1]-1)*l[0])/b
				cv2.line(img, (int(p[0]), int(p[1])), (int(img.shape[1]-1),int(point1)), (B,G,R), 2) 
			else:
				# point2 = (-l[2] - (a*200))/b
				cv2.line(img, (int(p[0]), int(p[1])), (int(point1),int(img.shape[0]-1)), (B,G,R), 2)
			print(l/l[2])
			print(p)
			print(int(img.shape[1]-1),int(point1))

	cv2.imshow('Three Lines', img)
	cv2.imwrite('Images/3SetsofLinesThroughVanishingPoint.jpg', img)
	cv2.waitKey(0)

############################ Part 5 : Affine Projection and Rectification ##################################

def part5(image):

	global vanish_line
	img = image.copy()
	# print(image.shape)
	src_point = np.array( [[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]] ).astype(np.float32)
	dst_point = np.array( [[0, img.shape[1]*0.33], [img.shape[1]*0.85, img.shape[0]*0.25], [img.shape[1]*0.15, img.shape[0]*0.7]] ).astype(np.float32)

	a = np.matmul(np.linalg.inv(np.append(src_point, np.ones((3,1)), axis = 1)), np.array([dst_point[0][0], dst_point[1][0], dst_point[2][0]]).reshape(-1, 1))
	b = np.matmul(np.linalg.inv(np.append(src_point, np.ones((3,1)), axis = 1)), np.array([dst_point[0][1], dst_point[1][1], dst_point[2][1]]).reshape(-1, 1))

	# vanish_line = part2(img)
	mat = np.transpose(np.append(a, b, axis=1))
	H = np.array([[1,0,0], [0,1,0]])
	H = np.append(H, vanish_line.reshape(1, -1), axis=0)  
	final_H  = np.matmul(mat, H)

	# for i in range(img.shape[0]):
	#   for j in range(img.shape[1]):
	output = cv2.warpAffine(img, final_H, (img.shape[1], img.shape[0]))
	cv2.imshow("Affine Transformation", output)
	cv2.imwrite('Images/AffineTransformation.jpg', output)
	cv2.waitKey(0)


if __name__ == '__main__':

	os.makedirs("Images", exist_ok = True)

	# Read image
	img = cv2.imread('Images/Garden.JPG')
	
	# Part 1 : GUI for drawing line segments
	# P2 = part1(img, 2)

	# Part 2 : Vanishing Line
	line = part2(img)

	# Part 3 : Parallel line to vanishing line
	part3(img, line)

	# Part 4 : Three Lines
	part4(img)

	# Part 5 : Affine rectification
	part5(img)