import numpy as np 

class LogisticRegressionSGD:

    # Inicialização do vetor de pesos w e w0 (bias)
    def weights(self, n_features):
        self.w = np.zeros((1,n_features))
        self.w0 = 0
        return self.w, self.w0
    
    # Cálculo do sigmóide
    def sigmoid(self, result):
        return 1/(1+np.exp(-result))
    
    # Cálculo da função custo J(w)
    def cost_func(self, w, w0, X, Y):
        m = X.shape[0]
        # Função custo
        cost = (-1/m)*(np.sum((Y.T*np.log(self.sigmoid(np.dot(w,X.T)+w0))) + ((1-Y.T)*(np.log(1-self.sigmoid(np.dot(w,X.T)+w0))))))
        # Gradientes
        dw = (1/m)*(np.dot(X.T, (self.sigmoid(np.dot(w,X.T)+w0)-Y.T).T))
        db = (1/m)*(np.sum(self.sigmoid(np.dot(w,X.T)+w0)-Y.T))
        g = (dw, db)
        
        return g, cost

    def model_predict(self, w, w0, X, Y, alfa, num):
        costs = []
        for i in range(num):
            g, cost = self.cost_func(w,w0,X,Y)
            dw = g[0]
            db = g[1]
            # Atualização dos pesos
            w = w - (alfa * (dw.T))
            w0 = w0 - (alfa * db)

            if (i % 500 == 0):
                costs.append(cost)
                        
        # Parâmetros finais
        coeff = (w, w0)
        gradient = (dw, db)

        return coeff, gradient, costs

    def predict(self, final_pred, m):
        y_pred = np.zeros((1,m))
        for i in range(final_pred.shape[1]):
            if final_pred[0][i] > 0.5:
                y_pred[0][i] = 1
                return y_pred