import cv2
import sys

if __name__ == '__main__':
	try:
		image_dir = sys.argv[1]
		mask_dir = sys.argv[2]
	except:
		exit(0)
	image = cv2.imread(image_dir)
	mask = cv2.imread(mask_dir)
	assert image.shape == mask.shape
	mask = mask[:, :, 0]
	image = cv2.inpaint(image, mask, 5, flags = cv2.INPAINT_NS)
	cv2.imwrite('background_inpainted.png', image)
