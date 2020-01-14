import cv2
import sys
import networkx as nx
import numpy as np
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline, interp1d
import munkres
import math

color_dist = .2

def pos2id(i, j, n, m):
	return i * m + j

def id2pos(id, n, m):
	return id // m, id % m

def get_log(d1, d2, n):
	l = [(10 ** (d1 + k * (d2 - d1) / (n - 1))) for k in xrange(0, n - 1)]
	l.append(10 ** d2)
	return l

def euclid_distance(p1, p2):
	return math.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))

def get_angle(p1, p2):
	return math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))

class shape_context(object):
	def __init__(self, bin_num = 5, theta = 12, rad_in = 0.125, rad_out = 2.0):
		self.bin_num = bin_num
		self.theta = theta
		self.rad_in = rad_in
		self.rad_out = rad_out
		self.total_bin = theta * bin_num
	
	def get_dist2(self, x, c):
		result = zeros((len(x), len(c)))
		for i in xrange(len(x)):
			for j in xrange(len(c)):
				result[i, j] = euclid_distance(x[i], c[j])
		return result
	
	def get_angle(self, x):
		result = zeros((len(x), len(x)))
		for i in xrange(len(x)):
			for j in xrange(len(x)):
				result[i, j] = get_angle(x[i], x[j])
		return result
	
	def chi_cost(self, hi, hj):
		cost = 0
		for k in xrange(self.total_bin):
			if (hi[k] + hj[k] > 0):
				cost += ((hi[k] - hj[k]) * (hi[k] - hj[k])) / (hi[k] + hj[k])
		return cost / 2.
	
	def compute(self, points, r = None):
		r_array = self.get_dist2(points, points)
		mean_dist = r_array.mean()
		r_array_n = r_array / mean_dist
		r_bin_edges = get_log(log10(self.rad_in), log10(self.rad_out), self.bin_num)  
		r_array_q = zeros((len(points), len(points)), dtype = int)
		for m in xrange(self.bin_num):
		   r_array_q += (r_array_n < r_bin_edges[m])
		fz = r_array_q > 0
		theta_array = self.get_angle(points)
		theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
		theta_array_q = 1 + floor(theta_array_2 / (2 * math.pi / self.theta))

		bins = zeros((len(points), self.total_bin))
		for i in xrange(len(points)):
			sn = zeros((self.bin_num, self.theta))
			for j in xrange(len(points)):
				if (fz[i, j]):
					sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
			bins[i] = sn.reshape(self.total_bin)
		return bins
	
	def cost(self, P, Q, qlength = None):
		p, tt = P.shape
		p2, tt = Q.shape
		d = p2
		if not qlength is None:
			d = qlength
		cost_matrix = zeros((p,p2))
		for i in xrange(p):
			for j in xrange(p2):
				cost_matrix[i,j] = self.chi_cost(Q[j] / d, P[i] / p)	
		return cost_matrix
		
	def hungurian(self, cost_matrix):
		m = munkres.Munkres()
		index = m.compute(cost_matrix.tolist())
		total = 0
		for row, column in index:
			value = cost_matrix[row][column]
			total += value
		return total, index
		
	def diff(self, p, q, qlength = None, method = 1):
		cost_matrix = self.cost(p, q, qlength)
		result = self.hungurian(cost_matrix)
		return result

	def interpolate(self, p1, p2):
		x = [0] * len(p1)
		xs = [0] * len(p1)
		y = [0] * len(p1)
		ys = [0] * len(p1)
		for i in xrange(len(p1)):
			x[i] = p1[i][0]
			xs[i] = p2[i][0]
			y[i] = p1[i][1]
			ys[i] = p2[i][1]	
		
		def func(r):
			res = r * r * log(r * r)
			res[r == 0] = 0
			return res
		
		smooth_const = 0.01	  
		fx = Rbf(x, xs, function = func, smooth = smooth_const)
		fy = Rbf(y, ys, function = func, smooth = smooth_const)
		cx, cy, e, cost, L = bookenstain(p1, p2, 15)
		return fx, fy, e, float(cost)

def complete(image1, mask1, image2, mask2):
	kappa = 1
	sigma = 1e2
	n, m = image1.shape[0], image1.shape[1]
	size = n * m
	flow_graph = nx.DiGraph()
	source = size
	target = source + 1
	for i in range(n):
		for j in range(m):
			image1[i, j, :] *= mask1[i, j, 0]
			image2[i, j, :] *= mask2[i, j, 0]
			if i > 0:
				delta = np.array(image1[i, j]) - np.array(image2[i - 1, j])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i - 1, j, n, m), capacity = wt)
			if i < n - 1:
				delta = np.array(image1[i, j]) - np.array(image2[i + 1, j])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i + 1, j, n, m), capacity = wt)
			if j > 0:
				delta = np.array(image1[i, j]) - np.array(image2[i, j - 1])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i, j - 1, n, m), capacity = wt)
			if j < m - 1:
				delta = np.array(image1[i, j]) - np.array(image2[i, j + 1])
				wt = kappa * math.exp(-1. * np.sum(np.dot(delta, delta)) / sigma)
				flow_graph.add_edge(pos2id(i, j, n, m), pos2id(i, j + 1, n, m), capacity = wt)
			if mask1[i, j, 0] < 1e-3 and mask2[i, j, 0] < 1e-3:
				continue
			if mask1[i, j, 0] < 1e-3:
				flow_graph.add_edge(source, pos2id(i, j, n, m), capacity = 1e-10)
				flow_graph.add_edge(pos2id(i, j, n, m), target, capacity = 1e10)
				continue
			if (abs(image1[i, j, 0] - image2[i, j, 0]) + abs(image1[i, j, 1] - image2[i, j, 1]) + abs(image1[i, j, 2] - image2[i, j, 2]) < color_dist):
				flow_graph.add_edge(pos2id(i, j, n, m), target, capacity = 1e-10)
				flow_graph.add_edge(source, pos2id(i, j, n, m), capacity = 1e10)
	cut_value, partition = nx.minimum_cut(flow_graph, source, target)
	reachable, non_reachable = partition
	res = np.zeros((n, m))
	for node in reachable:
		if node < source:
			res[id2pos(node, n, m)] = 1
	for node in non_reachable:
		if node < source:
			res[id2pos(node, n, m)] = -1
	output_image = np.zeros((n, m, 3))
	for i in range(n):
		for j in range(m):
			if mask1[i, j, 0] < 1e-3 and mask2[i, j, 0] < 1e-3:
				continue
			if res[i, j] == 1:
				output_image[i, j, 0] = image1[i, j, 0]
				output_image[i, j, 1] = image1[i, j, 1]
				output_image[i, j, 2] = image1[i, j, 2]
			else:
				output_image[i, j, 0] = image2[i, j, 0]
				output_image[i, j, 1] = image2[i, j, 1]
				output_image[i, j, 2] = image2[i, j, 2]
	return output_image

if __name__ == '__main__':
	try:
		image1_dir = sys.argv[1]
		mask1_dir = sys.argv[2]
		image2_dir = sys.argv[3]
		mask2_dir = sys.argv[4]
		output_dir = sys.argv[5]
	except:
		exit(0)
	image1 = cv2.imread(image1_dir)
	mask1 = cv2.imread(mask1_dir)
	image2 = cv2.imread(image2_dir)
	mask2 = cv2.imread(mask2_dir)
	image1 = image1 / 255.
	mask1 = mask1 / 255.
	image2 = image2 / 255.
	mask2 = mask2 / 255.
	output = complete(image1, mask1, image2, mask2)
	output = (np.clip(output, 0., 1.) * 255).astype(np.uint8)
	cv2.imwrite(output_dir, output)
