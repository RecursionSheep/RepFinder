import cv2
import sys
import numpy as np
import queue

score_threshold = .49
local_width = 5
matting_width = 4

def pattern_match(image, stroke, bbm, extended_bbm):
	n, m = image.shape[0], image.shape[1]
	size = n * m
	bbm = bbm / 255.
	extended_bbm = extended_bbm / 255.
	for i in range(n):
		for j in range(m):
			for k in range(1, 3):
				bbm[i, j, k] = (bbm[i, j, k] - .5) * 2
				extended_bbm[i, j, k] = (extended_bbm[i, j, k] - .5) * 2
				bbm[i, j, k] *= bbm[i, j, 0]
				extended_bbm[i, j, k] *= extended_bbm[i, j, 0]
	x = y = -1
	for i in range(n):
		for j in range(m):
			if abs(image[i, j, 0] - stroke[i, j, 0]) <= 5e-2 or abs(image[i, j, 1] - stroke[i, j, 1]) <= 5e-2 or abs(image[i, j, 2] - stroke[i, j, 2]) <= 5e-2:
				continue
			if stroke[i, j, 2] > stroke[i, j, 1]:
				x, y = i, j
				break
		if x != -1:
			break
	q = queue.Queue()
	q.put([x, y])
	visited = np.zeros((n, m))
	visited[x, y] = 1
	while not q.empty():
		[x, y] = q.get()
		if bbm[x, y, 0] > 1e-3:
			while not q.empty():
				q.get()
			break
		if x > 0 and visited[x - 1, y] == 0:
			visited[x - 1, y] = 1
			q.put([x - 1, y])
		if y > 0 and visited[x, y - 1] == 0:
			visited[x, y - 1] = 1
			q.put([x, y - 1])
		if x < n - 1 and visited[x + 1, y] == 0:
			visited[x + 1, y] = 1
			q.put([x + 1, y])
		if y < m - 1 and visited[x, y + 1] == 0:
			visited[x, y + 1] = 1
			q.put([x, y + 1])
	visited[:, :] = 0
	q.put([x, y])
	visited[x, y] = 1
	pattern = []
	while not q.empty():
		[x, y] = q.get()
		pattern.append([x, y])
		if x > 0 and visited[x - 1, y] == 0 and bbm[x - 1, y, 0] > 1e-3:
			visited[x - 1, y] = 1
			q.put([x - 1, y])
		if y > 0 and visited[x, y - 1] == 0 and bbm[x, y - 1, 0] > 1e-3:
			visited[x, y - 1] = 1
			q.put([x, y - 1])
		if x < n - 1 and visited[x + 1, y] == 0 and bbm[x + 1, y, 0] > 1e-3:
			visited[x + 1, y] = 1
			q.put([x + 1, y])
		if y < m - 1 and visited[x, y + 1] == 0 and bbm[x, y + 1, 0] > 1e-3:
			visited[x, y + 1] = 1
			q.put([x, y + 1])
	minx = miny = size
	maxx = maxy = 0
	for [x, y] in pattern:
		minx = min(minx, x)
		maxx = max(maxx, x)
		miny = min(miny, y)
		maxy = max(maxy, y)
	'''template = np.zeros((n, m, 3))
	template[minx : maxx + 1, miny : maxy + 1, :] = image[minx : maxx + 1, miny : maxy + 1, :]
	template = (np.clip(template, 0., 1.) * 255).astype(np.uint8)
	cv2.imwrite('template.png', template)'''
	pattern_n, pattern_m = maxx - minx + 1, maxy - miny + 1
	
	score = np.zeros((n, m))
	max_score = 0
	min_score = 1e100
	partial_sum = np.zeros((n, m))
	print('matching')
	for i in range(n):
		for j in range(m):
			if i > 0:
				partial_sum[i, j] += partial_sum[i - 1, j]
			if j > 0:
				partial_sum[i, j] += partial_sum[i, j - 1]
			if i > 0 and j > 0:
				partial_sum[i, j] -= partial_sum[i - 1, j - 1]
			if extended_bbm[i, j, 0] > 1e-3:
				partial_sum[i, j] += 1
	for i in range(n - pattern_n):
		for j in range(m - pattern_m):
			if partial_sum[i + pattern_n - 1, j + pattern_m - 1] - partial_sum[i, j + pattern_m - 1] - partial_sum[i + pattern_n - 1, j] + partial_sum[i, j] == 0:
				continue
			for [x, y] in pattern:
				vec1 = [bbm[x, y, 1], bbm[x, y, 2]]
				vec2 = [extended_bbm[i + x - minx, j + y - miny, 1], extended_bbm[i + x - minx, j + y - miny, 2]]
				score[i, j] += vec1[0] * vec2[0] + vec1[1] * vec2[1]
			max_score = max(max_score, score[i, j])
			min_score = min(min_score, score[i, j])
	score = (score - min_score) / (max_score - min_score)
	object_num = 0
	object_list = []
	visited = np.zeros((n, m))
	print('finding objects')
	trimap = np.zeros((pattern_n, pattern_m, 3))
	for i in range(n - pattern_n):
		for j in range(m - pattern_m):
			if visited[i, j] == 1:
				continue
			if score[i, j] < score_threshold:
				continue
			flag = True
			for dx in range(-local_width, local_width + 1):
				if i + dx < 0 or i + dx >= n:
					continue
				for dy in range(-local_width, local_width + 1):
					if j + dy < 0 or j + dy >= m:
						continue
					if score[i, j] < score[i + dx, j + dy]:
						flag = False
						break
					visited[i + dx, j + dy] = 1
				if not flag:
					break
			if flag:
				minx, miny = i, j
				maxx, maxy = i + pattern_n - 1, j + pattern_m - 1
				if score[i, j] > 1 - 1e-3:
					print('template', score[i, j])
					trimap = image[i : i + pattern_n, j : j + pattern_m, :].copy()
					for i0 in range(pattern_n):
						for j0 in range(pattern_m):
							x, y = i + i0, j + j0
							flag = True
							for dx in range(-matting_width, matting_width + 1):
								for dy in range(-matting_width, matting_width + 1):
									if x + dx < 0 or x + dx >= n or y + dy < 0 or y + dy >= m:
										continue
									if bbm[x + dx, y + dy, 0] > 1e-3:
										flag = False
										break
								if not flag:
									break
							if not flag:
								continue
							yy = y
							while yy >= miny:
								if bbm[x, yy, 0] > 1e-3:
									break
								yy -= 1
							if yy < miny:
								trimap[x - minx, y - miny, 0] = trimap[x - minx, y - miny, 1] = trimap[x - minx, y - miny, 2] = 0.
								continue
							yy = y
							while yy <= maxy:
								if bbm[x, yy, 0] > 1e-3:
									break
								yy += 1
							if yy > maxy:
								trimap[x - minx, y - miny, 0] = trimap[x - minx, y - miny, 1] = trimap[x - minx, y - miny, 2] = 0.
								continue
							trimap[x - minx, y - miny, 0] = trimap[x - minx, y - miny, 1] = trimap[x - minx, y - miny, 2] = 1.
					trimap = trimap * 255.
					trimap = trimap.astype(np.uint8)
					#cv2.imwrite('trimap%d.png' % object_num, trimap)
				object_list.append([i, j])
				object = image[i : i + pattern_n, j : j + pattern_m, :].copy()
				object = object * 255.
				object = object.astype(np.uint8)
				cv2.imwrite('object%d.png' % object_num, object)
				print("object", object_num, score[i, j])
				object_num += 1
	for id in range(object_num):
		[x, y] = object_list[id]
		trimap_out = image[x : x + pattern_n, y : y + pattern_m, :].copy()
		trimap_out = trimap_out * 255.
		trimap_out = trimap_out.astype(np.uint8)
		for i in range(pattern_n):
			for j in range(pattern_m):
				if trimap[i, j, 0] == 0 and trimap[i, j, 1] == 0 and trimap[i, j, 2] == 0:
					trimap_out[i, j, 0] = trimap_out[i, j, 1] = trimap_out[i, j, 2] = 0
				if trimap[i, j, 0] == 255 and trimap[i, j, 1] == 255 and trimap[i, j, 2] == 255:
					trimap_out[i, j, 0] = trimap_out[i, j, 1] = trimap_out[i, j, 2] = 255
		cv2.imwrite('trimap%d.png' % id, trimap_out)
	return object_list

if __name__ == '__main__':
	try:
		image_dir = sys.argv[1]
		stroke_dir = sys.argv[2]
		bbm_dir = sys.argv[3]
		extended_bbm_dir = sys.argv[4]
	except:
		exit(0)
	image = cv2.imread(image_dir)
	stroke = cv2.imread(stroke_dir)
	bbm = cv2.imread(bbm_dir)
	extended_bbm = cv2.imread(extended_bbm_dir)
	image = image / 255.
	stroke = stroke / 255.
	bbm = cv2.cvtColor(bbm, cv2.COLOR_BGR2Lab)
	extended_bbm = cv2.cvtColor(extended_bbm, cv2.COLOR_BGR2Lab)
	object_list = pattern_match(image, stroke, bbm, extended_bbm)
	objects = open("object_list.txt", "w")
	objects.write("%d\n" % len(object_list))
	for object in object_list:
		objects.write("%d %d\n" % (object[0], object[1]))
	objects.close()
