import numpy as np 
#from sklearn.utils.extmath import safe_sparse_dot
#from sklearn.metrics import mean_squared_error
import copy
import random as rd

class LogisticRegression:

#     def __init__(self, max_iter=100, eta0=0.0001):
#          self.iteracoes = max_iter
#          self.eta0 = eta0
#          self.hist = []

        
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

    def weightInitialization(self, n_features):
        self.w = np.zeros((1,n_features))
        self.w0 = 0
        return self.w, self.w0

    def sigmoid(self, result):
        return 1/(1+np.exp(-result))
    
    def cost_func(self, w, w0, X, Y):
        m = X.shape[0]
        # Função custo
        cost = (-1/m)*(np.sum((Y.T*np.log(self.sigmoid(np.dot(w,X.T)+w0))) + ((1-Y.T)*(np.log(1-self.sigmoid(np.dot(w,X.T)+w0))))))
        # Gradientes
        dw = (1/m)*(np.dot(X.T, (self.sigmoid(np.dot(w,X.T)+w0)-Y.T).T))
        db = (1/m)*(np.sum(self.sigmoid(np.dot(w,X.T)+w0)-Y.T))
        grads = (dw, db)
        
        return grads, cost


    def model_predict(self, w, w0, X, Y, alfa, num):
        costs = []
        for i in range(num):
            grads, cost = self.cost_func(w,w0,X,Y)
            #
            dw = grads[0]
            db = grads[1]
            #weight update
            w = w - (alfa * (dw.T))
            w0 = w0 - (alfa * db)
            #
            if (i % 100 == 0):
                costs.append(cost)
                #print("Interação: ",i)
                #print("Valor do custo:",cost)
                
        
        #final parameters
        coeff = (w, w0)
        #coeff = {"w": w, "w0": w0}
        gradient = (dw, db)
        #gradient = {"dw": dw, "db": db}
        return coeff, gradient, costs

    def predict(self, final_pred, m):
        y_pred = np.zeros((1,m))
        for i in range(final_pred.shape[1]):
            if final_pred[0][i] > 0.5:
                y_pred[0][i] = 1
                return y_pred