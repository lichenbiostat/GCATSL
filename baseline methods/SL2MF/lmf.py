import pdb
import numpy as np


class LMF:
    def __init__(self, num_factors=10, nn_size=100, theta=1.0, reg=0.01, alpha=0.01, beta=0.01, beta1=0.01, beta2=0.01, max_iter=30, seed=123):
        self.num_factors = num_factors
        self.nn_size = nn_size
        self.theta = theta
        self.reg = reg
        self.alpha = alpha
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.max_iter = max_iter
        self.seed = seed

    def deriv(self):
        vec_deriv = -np.dot(self.weight_IntMat, self.U)
        A = np.dot(self.U, self.U.T)
        A = np.exp(A)
        A /= (A + self.ones)
        A = self.W * A
        vec_deriv += np.dot(A, self.U)
        vec_deriv += self.reg*self.U
        if self.alpha > 0 and self.GoSim is not None:
            vec_deriv += self.alpha*np.dot(self.GoLap, self.U)
        if self.beta > 0 and self.GoCCSim is not None:
            vec_deriv += self.beta*np.dot(self.CCLap, self.U)
        if self.beta1 > 0 and self.PPISim is not None:
            vec_deriv += self.beta1*np.dot(self.PPILap, self.U)
        #if self.beta2 > 0 and self.COPATHSim is not None:
        #    vec_deriv += self.beta2*np.dot(self.PATHLap, self.U)    
        return vec_deriv
    
    def deriv_long(self):
        #vec_deriv = -np.dot(self.weight_IntMat, self.U)
        P = np.dot(self.U, self.U.T)
        P = np.exp(P)
        P /= (P + self.ones)
        P = P * self.mask
        self.IntMat = self.IntMat * self.mask
        P_M = P - self.IntMat
        P_M = P_M * self.W
        vec_deriv += np.dot(P_M, self.U)

        vec_deriv += self.reg*self.U
        if self.alpha > 0 and self.GoSim is not None:
            vec_deriv += self.alpha*np.dot(self.GoLap, self.U)
        if self.beta > 0 and self.GoCCSim is not None:
            vec_deriv += self.beta*np.dot(self.CCLap, self.U)
        if self.beta1 > 0 and self.PPISim is not None:
            vec_deriv += self.beta1*np.dot(self.PPILap, self.U)
        #if self.beta2 > 0 and self.COPATHSim is not None:
        #    vec_deriv += self.beta2*np.dot(self.PATHLap, self.U)    
        return vec_deriv

    def compute_loss(self):
        A = np.dot(self.U, self.U.T)
        B = A * self.weight_IntMat
        B = B * self.mask  
        loss = -np.sum(B)  
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.W * A
        A = A * self.mask 
        loss += np.sum(A)   
        loss = 0.5*loss+0.5*self.reg*np.sum(np.square(self.U))
        if self.alpha > 0 and self.GoSim is not None:
            loss += 0.5*self.alpha*np.sum(np.diag((np.dot(self.U.T, self.GoLap)).dot(self.U)))
        if self.beta > 0 and self.GoCCSim is not None:
            loss += 0.5*self.beta*np.sum(np.diag((np.dot(self.U.T, self.CCLap)).dot(self.U)))
        if self.beta1 > 0 and self.PPISim is not None:
            loss += 0.5*self.beta1*np.sum(np.diag((np.dot(self.U.T, self.PPILap)).dot(self.U))) 
        #if self.beta2 > 0 and self.COPATHSim is not None:
        #    loss += 0.5*self.beta2*np.sum(np.diag((np.dot(self.U.T, self.PATHLap)).dot(self.U)))    
        return loss

    def build_KNN_matrix(self, S, nn_size):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(nn_size, n)]
            X[i, ii] = S[i, ii]
        return X

    def compute_laplacian_matrix(self, S, nn_size):
        if nn_size > 0:
            S1 = self.build_KNN_matrix(S, nn_size)
            x = np.sum(S1, axis=1)
            L = np.diag(x) - S1
        else:
            x = np.sum(S, axis=1)
            L = np.diag(x) - S
        return L

    def fix(self, IntMat, W, mask, GoSim=None, GoCCSim=None, PPISim=None, COPATHSim=None):
        '''
        IntMat: The sparse interaction matrix
        W: the weighting matrix
        GoSim: the GO similarity matrix
        TpSim: the topology structure similarity matrix
        '''
        self.IntMat, self.W, self.mask = IntMat, W, mask
        self.GoSim, self.GoCCSim, self.PPISim, self.COPATHSim = GoSim, GoCCSim, PPISim, COPATHSim
        self.weight_IntMat = self.IntMat*self.W
        if self.alpha > 0 and self.GoSim is not None:
            self.GoLap = self.compute_laplacian_matrix(self.GoSim, self.nn_size)
        if self.beta > 0 and self.GoCCSim is not None:
            self.CCLap = self.compute_laplacian_matrix(self.GoCCSim, self.nn_size)
        if self.beta1 > 0 and self.PPISim is not None:
            self.PPILap = self.compute_laplacian_matrix(self.PPISim, self.nn_size)
        #if self.beta2 > 0 and self.COPATHSim is not None:
        #    self.PATHLap = self.compute_laplacian_matrix(self.COPATHSim, self.nn_size)    
            
        self.num_rows = IntMat.shape[0]
        self.ones = np.ones((self.num_rows, self.num_rows))
        prng = np.random.RandomState(self.seed)
        self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_rows, self.num_factors))
        grad_sum = np.zeros((self.num_rows, self.num_factors))
        last_log = self.compute_loss()
        for t in range(self.max_iter):
            print("iteration: %d" % t)
            grad = self.deriv()
            grad_sum += np.square(grad)
            vec_step_size = self.theta / np.sqrt(grad_sum)
            self.U -= vec_step_size * grad
            curr_log = self.compute_loss()
            delta_log = (curr_log-last_log)/abs(last_log)
            # print "iter:%s, curr_loss:%s, last_loss:%s, delta_loss:%s" % (t, curr_log, last_log, delta_log)
            if abs(delta_log) < 1e-5:
                break
            last_log = curr_log
        # print "complete model training"

    def smooth_prediction(self, test_data):
        pass


    def predict(self, test_data):
        val = np.sum(self.U[test_data[:, 0], :]*self.U[test_data[:, 1], :], axis=1)
        val = np.exp(val)
        val = val/(1+val)
        return val

    def __str__(self):
        return "Model: LMF, num_factors:%s, nn_size:%s, theta:%s, reg:%s, alpha:%s, beta:%s, max_iter:%s, seed:%s" % (self.num_factors, self.nn_size, self.theta, self.reg, self.alpha, self.beta, self.max_iter, self.seed)
