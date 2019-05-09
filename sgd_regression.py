import numpy as np 
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics import mean_squared_error
import copy
import random as rd

class SGDRegression:

	def __init__(self, max_iter=100, eta0=0.0001):

		self.iteracoes = max_iter
		self.eta0 = eta0		
		self.hist = []		

	def fit(self, X, y):
		
		data = copy.deepcopy(X)
		linhas, colunas = data.shape 
		self.weightInitialization(colunas)



		for i in range(0, self.iteracoes):

			choice = rd.randint(0,linhas-1)

			elem = data[choice,:].reshape((1,colunas))
			y_true = y[choice].reshape((1,1))

			grads, cost = self.model_optimize(elem, y_true)

			dw = grads["dw"]
			db = grads["db"]
			#weight update
			self.w = self.w - ((self.eta0/linhas) * (dw.T))
			self.b = self.b - ((self.eta0/linhas) * db)	

			print("Iteration: " + str(i) + " " + "Loss: " + str(cost))

	def sigmoid_activation(self, result):
		return 1/(1+np.exp(-result))
		

	def weightInitialization(self, n_features):
		self.w = np.zeros((1,n_features))
		self.b = 0		

	def model_optimize(self, X, Y):
		m = X.shape[0]

		#Prediction
		final_result = self.sigmoid_activation(float(np.dot(self.w,X.T)+self.b))
		Y_T = Y.T
		cost = (-1/m)*(np.sum((Y_T*np.log(final_result + 1e-5)) + ((1-Y_T)*(np.log(1-final_result + 1e-5)))))
		#

		#Gradient calculation
		dw = (1/m)*cost*Y_T*final_result*(np.dot(X.T, final_result*(1-final_result).T))
		db = (1/m)*(np.sum(final_result-Y.T))

		grads = {"dw": dw, "db": db}

		return grads, cost