import cv2
import numpy as np

#################### Part 1 : GUI for obtaning 2D Projective Space Representation ################

global points
global vanish_line

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
	make_line = int(input('Press 1 for drawing line segments on image, 0 for exiting GUI :'))

	while make_line:
		print("Double click on 2 points on the image and then close the window")
		
		points = []	
		show = True
		while show:
			show = imshow('image', img, numLines)
		# print(points)
		cv2.line(img, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0,255,0), 2)

		for i in range(int(len(points)/2)):
			x1, y1, x2, y2 = float(points[2*i][0]), float(points[2*i][1]), float(points[2*i+1][0]), float(points[2*i+1][1])
			cv2.line(img, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 2)
			
			# Compute line parameters -> ax+by+c = 0
			# a = y2 - y1
			# b = x1 - x2
			# c = (x2-x1)*y1 - (y2-y1)*x1
			p = np.cross([x1,y1,1], [x2,y2,1])
			# print(p)
			p /= p[2]
			P2.append(p)

		make_line = int(input('Press 1 for drawing line segments on image, 0 for exiting GUI :'))

	cv2.imshow('image_part1', img)
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
	vanish_line = line
	point1 /= point1[2]
	point2 /= point2[2]
	
	cv2.line(img, (int(point1[0]), int(point1[1])), (int(point2[0]),int(point2[1])), (0,255,0), 2)

	cv2.imshow('vanishing line', img)
	cv2.waitKey(0)
	
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


	centre_line = np.cross(point1, point2)
	centre_line /= centre_line[2]

	cv2.imshow('parallel line', img)
	cv2.waitKey(0)
	

def part4(image):

	img = image.copy()
	p = [img.shape[1]//2, img.shape[0]//2, 1]

	H = np.random.rand(2,3) * 1000
	# H = np.array([[1,0,0],[0,1,0]])
	a = np.array([-p[1], p[0]-(1.0/p[1]) ,1]).astype(np.float32).reshape(1,-1)
	H = np.append(H, a, axis = 0)
	print(H)
	result_point = H @ np.transpose(p)

	projected_a = -result_point[1]
	projected_b = result_point[0]
	
	for c in range(1,4):
		l = np.transpose(H) @ np.array([projected_a, projected_b, c*1000]).reshape(-1, 1)
		a = l[0]
		b = l[1]
		point1 = (-l[2])/a
		print(l)
		print(point1)
		# print(p)
		# point2 = (-l[2] - (a*200))/b
		cv2.line(img, (int(p[0]), int(p[1])), (int(point1),int(0)), (255,0,0), 2)

	cv2.imshow('Three Lines', img)
	cv2.waitKey(0)

def part5(image):

	global vanish_line
	img = image.copy()
	# print(image.shape)
	src_point = np.array( [[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]] ).astype(np.float32)
	dst_point = np.array( [[0, img.shape[1]*0.33], [img.shape[1]*0.85, img.shape[0]*0.25], [img.shape[1]*0.15, img.shape[0]*0.7]] ).astype(np.float32)

	# mat = np.zeros((3,2))
	# print(np.append(src_point, np.ones((3,1)), axis = 1).shape)

	a = np.linalg.inv(np.append(src_point, np.ones((3,1)), axis = 1)) @ np.array([dst_point[0][0], dst_point[1][0], dst_point[2][0]]).reshape(-1, 1)
	b = np.linalg.inv(np.append(src_point, np.ones((3,1)), axis = 1)) @ np.array([dst_point[0][1], dst_point[1][1], dst_point[2][1]]).reshape(-1, 1)

	# print(a , b)
	print(np.transpose(np.append(a, b, axis=1)))
	
	# vanish_line = part2(img)
	mat = np.transpose(np.append(a, b, axis=1))
	H = np.array([[1,0,0], [0,1,0]])
	print(vanish_line)
	# vanish_line /= np.sqrt(np.sum(vanish_line ** 2))
	H = np.append(H, vanish_line.reshape(1, -1), axis=0)
	print("Vanish Line - : ", vanish_line)
	print("Rectifier Matrix - : ", H)
	print("Normal Matrix - : ", mat)
	final_H  = mat @ H
	print(final_H.shape)
	# for i in range(img.shape[0]):
	# 	for j in range(img.shape[1]):
	output = cv2.warpAffine(img, final_H, (img.shape[1], img.shape[0]))
	cv2.imshow("Affine Transformation", output)
	cv2.waitKey(0)




if __name__ == '__main__':

	# Read image
	img = cv2.imread('Garden.JPG')
	
	# Part 1 : GUI for drawing line segments
	# P2 = part1(img, 2)

	# Part 2 : Vanishing Line
	line = part2(img)

	# Part 3 : Parallel line to vanishing line
	# part3(img)

	# Part 4 : Three Lines
	# part4(img)

	# Part 5 : Affine rectification
	part5(img)