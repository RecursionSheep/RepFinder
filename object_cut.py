import cv2
import numpy as np
import sys
import math
import networkx as nx
import queue

threshold = 500

def pos2id(i, j, n, m):
	return i * m + j

def id2pos(id, n, m):
	return id // m, id % m

def GaussianMLE(points):
	mu = np.zeros(3)
	for point in points:
		mu += point
	mu /= len(points)
	
	sigma = np.zeros((3, 3))
	for point in points:
		x_delta = point - mu
		for i in range(3):
			for j in range(3):
				sigma[i, j] += x_delta[i] * x_delta[j]
	sigma /= len(points)
	return mu, sigma

def object_cut(image, stroke):
	kappa = 10
	sigma = 1e2
	n, m = image.shape[0], image.shape[1]
	size = n * m
	object = []
	background = []
	for i in range(n):
		for j in range(m):
			if abs(image[i, j, 0] - stroke[i, j, 0]) <= 1e-2 and abs(image[i, j, 1] - stroke[i, j, 1]) <= 1e-2 and abs(image[i, j, 2] - stroke[i, j, 2]) <= 1e-2:
				continue
			# red for object, green for background
			if stroke[i, j, 2] > stroke[i, j, 1]:
				object.append(np.array(image[i, j]))
			if stroke[i, j, 1] > stroke[i, j, 2]:
				background.append(np.array(image[i, j]))
	Mu1, Sigma1 = GaussianMLE(object)
	Mu2, Sigma2 = GaussianMLE(background)
	prob_object = np.zeros((n, m))
	prob_background = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			delta1 = (np.array(image[i, j]) - Mu1).reshape(3, 1)
			delta2 = (np.array(image[i, j]) - Mu2).reshape(3, 1)
			prob_object[i, j] = math.exp(-0.5 * np.matmul(np.matmul(np.transpose(np.array(image[i, j]) - Mu1), np.linalg.inv(Sigma1)), np.array(image[i, j]) - Mu1)) / math.sqrt(np.linalg.det(Sigma1))
			prob_background[i, j] = math.exp(-0.5 * np.matmul(np.matmul(np.transpose(np.array(image[i, j]) - Mu2), np.linalg.inv(Sigma2)), np.array(image[i, j]) - Mu2)) / math.sqrt(np.linalg.det(Sigma2))
			if abs(image[i, j, 0] - stroke[i, j, 0]) <= 1e-2 and abs(image[i, j, 1] - stroke[i, j, 1]) <= 1e-2 and abs(image[i, j, 2] - stroke[i, j, 2]) <= 1e-2:
				continue
			# red for object, green for background
			if stroke[i, j, 2] > stroke[i, j, 1]:
				prob_object[i, j] = 1
				prob_background[i, j] = 1e-10
			if stroke[i, j, 1] > stroke[i, j, 2]:
				prob_background[i, j] = 1
				prob_object[i, j] = 1e-10
	flow_graph = nx.DiGraph()
	source = n * m
	target = source + 1
	for i in range(n):
		for j in range(m):
			if (prob_object[i, j] < 1e-10):
				prob_object[i, j] = 1e-10
			if (prob_background[i, j] < 1e-10):
				prob_background[i, j] = 1e-10
			flow_graph.add_edge(pos2id(i, j, n, m), target, capacity = -math.log(prob_object[i, j] / (prob_object[i, j] + prob_background[i, j])))
			flow_graph.add_edge(source, pos2id(i, j, n, m), capacity = -math.log(prob_background[i, j] / (prob_object[i, j] + prob_background[i, j])))
			if i > 0:
				delta = np.array(image[i, j]) - np.array(image[i - 1, j])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i - 1, j, n, m), capacity = wt)
			if i < n - 1:
				delta = np.array(image[i, j]) - np.array(image[i + 1, j])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i + 1, j, n, m), capacity = wt)
			if j > 0:
				delta = np.array(image[i, j]) - np.array(image[i, j - 1])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i, j - 1, n, m), capacity = wt)
			if j < m - 1:
				delta = np.array(image[i, j]) - np.array(image[i, j + 1])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i, j + 1, n, m), capacity = wt)
	print("finish building graph")
	cut_value, partition = nx.minimum_cut(flow_graph, source, target)
	reachable, non_reachable = partition
	res = np.zeros((n, m))
	for node in reachable:
		if node < source:
			res[id2pos(node, n, m)] = 1
	for node in non_reachable:
		if node < source:
			res[id2pos(node, n, m)] = -1
	
	size = np.zeros((n, m))
	visited = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			if (visited[i, j] == 0):
				q = queue.Queue()
				cluster = []
				q.put((i, j))
				visited[i, j] = 1
				size[i, j] = 0
				while not q.empty():
					x, y = q.get()
					size[i, j] += 1
					cluster.append((x, y))
					if (x > 0 and visited[x - 1, y] == 0 and res[x - 1, y] == res[i, j]):
						q.put((x - 1, y))
						visited[x - 1, y] = 1
					if (y > 0 and visited[x, y - 1] == 0 and res[x, y - 1] == res[i, j]):
						q.put((x, y - 1))
						visited[x, y - 1] = 1
					if (x < n - 1 and visited[x + 1, y] == 0 and res[x + 1, y] == res[i, j]):
						q.put((x + 1, y))
						visited[x + 1, y] = 1
					if (y < m - 1 and visited[x, y + 1] == 0 and res[x, y + 1] == res[i, j]):
						q.put((x, y + 1))
						visited[x, y + 1] = 1
				if size[i, j] < threshold:
					for (x, y) in cluster:
						res[x, y] = - res[x, y]
	visited = np.zeros((n, m))
	for i in range(n):
		for j in range(m):
			if (visited[i, j] == 0):
				q = queue.Queue()
				cluster = []
				q.put((i, j))
				visited[i, j] = 1
				size[i, j] = 0
				while not q.empty():
					x, y = q.get()
					size[i, j] += 1
					cluster.append((x, y))
					if (x > 0 and visited[x - 1, y] == 0 and res[x - 1, y] == res[i, j]):
						q.put((x - 1, y))
						visited[x - 1, y] = 1
					if (y > 0 and visited[x, y - 1] == 0 and res[x, y - 1] == res[i, j]):
						q.put((x, y - 1))
						visited[x, y - 1] = 1
					if (x < n - 1 and visited[x + 1, y] == 0 and res[x + 1, y] == res[i, j]):
						q.put((x + 1, y))
						visited[x + 1, y] = 1
					if (y < m - 1 and visited[x, y + 1] == 0 and res[x, y + 1] == res[i, j]):
						q.put((x, y + 1))
						visited[x, y + 1] = 1
				if size[i, j] < threshold:
					for (x, y) in cluster:
						res[x, y] = -res[x, y]
	print('finish BFS')
	
	object_image = np.zeros((n, m, 3))
	background_image = np.zeros((n, m, 3))
	mask = np.zeros((n, m, 3))
	for i in range(n):
		for j in range(m):
			if (res[i, j] == 1):
				object_image[i, j] = image[i, j]
				mask[i, j, 0] = mask[i, j, 1] = mask[i, j, 2] = 1.
			if (res[i, j] == -1):
				background_image[i, j] = image[i, j]
	return object_image, background_image, mask

if __name__ == '__main__':
	try:
		image_dir = sys.argv[1]
		stroke_dir = sys.argv[2]
	except:
		exit(0)
	image = cv2.imread(image_dir)
	stroke = cv2.imread(stroke_dir)
	assert image.shape == stroke.shape
	image = image / 255.0
	stroke = stroke / 255.0
	object, background, mask = object_cut(image, stroke)
	print('finish graph cut')
	object = (np.clip(object, 0., 1.) * 255).astype(np.uint8)
	background = (np.clip(background, 0., 1.) * 255).astype(np.uint8)
	mask = (np.clip(mask, 0., 1.) * 255).astype(np.uint8)

	cv2.imwrite('object.png', object)
	cv2.imwrite('background.png', background)
	cv2.imwrite('mask.png', mask)
