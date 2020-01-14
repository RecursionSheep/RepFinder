import cv2
import numpy as np
import queue
import math
import sys

extend_len = 4
band_width = 2

def is_boundary(x, y, image):
	if image[x, y, 0] > 5e-3 or image[x, y, 1] > 5e-3 or image[x, y, 2] > 5e-3:
		return True
	return False

def extend_boundary(image):
	'''
	output is Lab-image
	'''
	n, m = image.shape[0], image.shape[1]
	size = n * m
	output = np.zeros((n, m, 3))
	extended = np.zeros((n, m, 3))
	visited = np.zeros((n, m))
	q = queue.Queue()
	remained = 0
	for i in range(n):
		for j in range(m):
			if not is_boundary(i, j, image):
				output[i, j, 0] = 0
				output[i, j, 1] = output[i, j, 2] = 128
				continue
			remained += 1
			visited[i, j] = 1
			p = [0, 0]
			for dx in range(-extend_len, extend_len + 1):
				for dy in range(-extend_len, extend_len + 1):
					if dx == 0 and dy == 0:
						continue
					if (i + dx < 0 or i + dx >= n or j + dy < 0 or j + dy >= m):
						continue
					if not is_boundary(i + dx, j + dy, image):
						continue
					if dy < 0:
						p[0] += -dx / math.sqrt(dx * dx + dy * dy)
						p[1] += -dy / math.sqrt(dx * dx + dy * dy)
					else:
						p[0] += dx / math.sqrt(dx * dx + dy * dy)
						p[1] += dy / math.sqrt(dx * dx + dy * dy)
			t = math.sqrt(p[0] * p[0] + p[1] * p[1])
			p[0] /= t
			p[1] /= t
			output[i, j, 0] = image[i, j, 0] * 255.
			output[i, j, 1] = int(p[0] * 128.) + 128.
			output[i, j, 2] = int(p[1] * 128.) + 128.
			extended[i, j, 0] = image[i, j, 0]
			extended[i, j, 1] = p[0] * extended[i, j, 0]
			extended[i, j, 2] = p[1] * extended[i, j, 0]
			if i > 0 and not is_boundary(i - 1, j, image):
				q.put([i - 1, j])
			if j > 0 and not is_boundary(i, j - 1, image):
				q.put([i, j - 1])
			if i < n - 1 and not is_boundary(i + 1, j, image):
				q.put([i + 1, j])
			if j < m - 1 and not is_boundary(i, j + 1, image):
				q.put([i, j + 1])
	remained *= band_width
	while remained > 0 and not q.empty():
		i, j = q.get()
		if visited[i, j] == 1:
			continue
		visited[i, j] = 1
		cnt = 0
		neigh = [[i - 1, j], [i, j - 1], [i + 1, j], [i, j + 1]]
		for p in neigh:
			x, y = p[0], p[1]
			if not (x >= 0 and y >= 0 and x < n and y < m and visited[x, y] == 1):
				continue
			cnt += 1
			vec_sum = [extended[i, j, 1] + extended[x, y, 1], extended[i, j, 2] + extended[x, y, 2]]
			vec_diff = [extended[i, j, 1] - extended[x, y, 1], extended[i, j, 2] - extended[x, y, 2]]
			if (vec_sum[0] * vec_sum[0] + vec_sum[1] * vec_sum[1]) > (vec_diff[0] * vec_diff[0] + vec_diff[1] * vec_diff[1]):
				extended[i, j, 1] += extended[x, y, 1]
				extended[i, j, 2] += extended[x, y, 2]
			else:
				extended[i, j, 1] -= extended[x, y, 1]
				extended[i, j, 2] -= extended[x, y, 2]
		extended[i, j, 1] /= cnt
		extended[i, j, 2] /= cnt
		if (extended[i, j, 2] < 0):
			extended[i, j, 1] *= -1
			extended[i, j, 2] *= -1
		for p in neigh:
			x, y = p[0], p[1]
			if (x >= 0 and y >= 0 and x < n and y < m and visited[x, y] == 0):
				q.put([x, y])
		remained -= 1
	
	for i in range(n):
		for j in range(m):
			if visited[i, j] == 1:
				extended[i, j, 0] = math.sqrt(extended[i, j, 1] * extended[i, j, 1] + extended[i, j, 2] * extended[i, j, 2])
				extended[i, j, 1] /= extended[i, j, 0]
				extended[i, j, 2] /= extended[i, j, 0]
				if (extended[i, j, 0] > 1.):
					extended[i, j, 0] = 1.
			extended[i, j, 0] = extended[i, j, 0] * 255.
			extended[i, j, 1] = int(extended[i, j, 1] * 128.) + 128.
			extended[i, j, 2] = int(extended[i, j, 2] * 128.) + 128.
	
	output = output.astype(np.uint8)
	extended = extended.astype(np.uint8)
	output = cv2.cvtColor(output, cv2.COLOR_Lab2BGR)
	extended = cv2.cvtColor(extended, cv2.COLOR_Lab2BGR)
	return output, extended

if __name__ == '__main__':
	try:
		image_dir = sys.argv[1]
	except:
		exit(0)
	image = cv2.imread(image_dir)
	image = image / 255.0
	BBM, extended_BBM = extend_boundary(image)
	cv2.imwrite('BBM.png', BBM)
	cv2.imwrite('extended_BBM.png', extended_BBM)
