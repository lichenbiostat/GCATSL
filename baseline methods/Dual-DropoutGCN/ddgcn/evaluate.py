from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
import numpy as np
from scipy import stats
from scipy.stats import t


class Evaluator:
    def __init__(self, train_adj, test_adj=None, pos_threshold=None):   
        node_num = train_adj.shape[0]
        eval_x, eval_y = np.triu_indices(node_num, k=1)
        train_edge_x, train_edge_y = train_adj.nonzero()   
        
        #using all negative samples for test
        #eval_coord = set(zip(eval_x, eval_y)) - set(zip(train_edge_x, train_edge_y))
        #self.eval_coord = np.array(list(eval_coord))
        
     
        test_edge_x, test_edge_y = test_adj.nonzero()   
        eval_coord = set(zip(eval_x, eval_y)) - set(zip(train_edge_x, train_edge_y)) - set(zip(test_edge_x, test_edge_y))
        coord_negative = np.array(list(eval_coord))   
        
        reorder = np.arange(coord_negative.shape[0])
        np.random.shuffle(reorder)
        index_neg = reorder[0:test_edge_x.size]  
        
        test_edge_x_neg = coord_negative[index_neg,0]
        test_edge_y_neg = coord_negative[index_neg,1]   
        
        test_edge_x = np.append(test_edge_x, test_edge_x_neg, axis=0) 
        test_edge_y = np.append(test_edge_y, test_edge_y_neg, axis=0)  
        self.eval_coord = np.append(test_edge_x.reshape([-1,1]), test_edge_y.reshape([-1,1]), axis=1)
     
        
        if test_adj is not None:
            self.y_true = test_adj[self.eval_coord[:, 0], self.eval_coord[:, 1]]

        self.pos_threshold = pos_threshold

    @staticmethod
    def geometric_mean(reconstruct_adj1, reconstruct_adj2, rho):
        reconstruct_adj = np.power(reconstruct_adj1 * np.power(reconstruct_adj2, rho), 1/(1+rho))  

        return reconstruct_adj

    def eval(self, reconstruct_adj1, reconstruct_adj2, rho):
        reconstruct_adj = self.geometric_mean(reconstruct_adj1, reconstruct_adj2, rho)
        y_score = reconstruct_adj[self.eval_coord[:, 0], self.eval_coord[:, 1]]

        auc_test = roc_auc_score(self.y_true, y_score)
        precision, recall, thresholds = precision_recall_curve(self.y_true, y_score)
        aupr_test = auc(recall, precision)

        f1_test = f1_score(self.y_true, y_score > self.pos_threshold)

        return auc_test, aupr_test, f1_test

    def unknown_pairs_scores(self, reconstruct_adj1, reconstruct_adj2, rho):
        """
        :return y_score:1 D Tensor, e.g., tensor([ 1.0704,  0.6944, -0.5432])
        """
        reconstruct_adj = self.geometric_mean(reconstruct_adj1, reconstruct_adj2, rho)
        y_score = reconstruct_adj[self.eval_coord[:, 0], self.eval_coord[:, 1]]  
        return self.eval_coord, y_score


def cal_confidence_interval(data, confidence=0.95):
    data = 1.0*np.array(data)
    n = len(data)
    sample_mean = np.mean(data)
    se = stats.sem(data)
    t_ci = t.ppf((1+confidence)/2., n-1)  
    bound = se * t_ci
    return sample_mean, bound
