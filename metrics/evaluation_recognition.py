import math
import numpy as np

class Evaluation:

	def my_compute_rank1(self, predictions_arr, actualClasses):
		score=0
		for i in range(len(predictions_arr)):
			#print((np.argmax(predictions_arr[i]), actualClasses[i]))
			if (np.argmax(predictions_arr[i]) + 1) == actualClasses[i]:
				score+=1

		print(score)
		rank1 = score/float(len(actualClasses))
		return rank1*100

	def my_compute_rank5(self, predictions_arr, actualClasses):
		score=0
		for i in range(len(predictions_arr)):
			top5 = np.argpartition(predictions_arr[i][0], -5)[-5:]
			top5 = [x+1 for x in top5]
			if actualClasses[i] in top5:
				score+=1

		rank1 = score/float(len(actualClasses))
		return rank1*100

	def my_compute_rankN(self, predictions_arr, actualClasses, N):
		score=0
		for i in range(len(predictions_arr)):
			topN = np.argpartition(predictions_arr[i][0], -N)[-N:]
			topN = [x+1 for x in topN]
			if actualClasses[i] in topN:
				score+=1

		rank1 = score/float(len(actualClasses))
		return rank1*100

	def compute_rank1(self, Y, y):
		classes = np.unique(sorted(y))
		count_all = 0
		count_correct = 0
		for cla1 in classes:
			idx1 = y==cla1
			if (list(idx1).count(True)) <= 1:
				continue
			# Compute only for cases where there is more than one sample:
			Y1 = Y[idx1==True, :]
			Y1[Y1==0] = math.inf
			for y1 in Y1:
				s = np.argsort(y1)
				smin = s[0]
				imin = idx1[smin]
				count_all += 1
				if imin:
					count_correct += 1
		return count_correct/count_all*100


	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

		# def compute_rank5(self, Y, y):
	# 	# First loop over classes in order to select the closest for each class.
	# 	classes = np.unique(sorted(y))
		
	# 	sentinel = 0
	# 	for cla1 in classes:
	# 		idx1 = y==cla1
	# 		if (list(idx1).count(True)) <= 1:
	# 			continue
	# 		Y1 = Y[idx1==True, :]

	# 		for cla2 in classes:
	# 			# Select the closest that is higher than zero:
	# 			idx2 = y==cla2
	# 			if (list(idx2).count(True)) <= 1:
	# 				continue
	# 			Y2 = Y1[:, idx1==True]
	# 			Y2[Y2==0] = math.inf
	# 			min_val = np.min(np.array(Y2))
	# 			# ...