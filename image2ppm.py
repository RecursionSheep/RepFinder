import cv2
import sys

if __name__ == '__main__':
	try:
		image_dir = sys.argv[1]
	except:
		exit(0)
	image = cv2.imread(image_dir)
	cv2.imwrite('image.ppm', image)
