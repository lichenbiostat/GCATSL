

import pdb
import BPR
import time
import random
import numpy as np
from collections import defaultdict
# from functions import nDCGAtN


class BPRMF:

    def __init__(self, num_factors=100, reg_u=0.015, reg_i=0.015, theta=0.05,
                 max_iter=100, seed=123, outf=1):
        self.d = num_factors
        self.theta = theta
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.max_iter = max_iter
        self.seed = seed
        self.outf = outf

    def random_sample_triples(self, train_data, Tr_neg):
        pair_arr = [(u, v, BPR.uniform_random_id(Tr_neg[u]["items"], Tr_neg[u]["num"])) for u, v in train_data]
        return np.array(pair_arr, dtype=np.int32)

    def sample_validation_data(self, Tr, Tr_neg, ratio=0.05):
        validation = []
        num = max(int(np.ceil(ratio*self.num_users)), 100)
        sub_set = random.sample(Tr.keys(), num)
        for u in sub_set:
            validation.extend([(u, i, BPR.uniform_random_id(
                Tr_neg[u]["items"], Tr_neg[u]["num"])) for i in Tr[u]])
        return np.array(validation)

    def fix_model(self, train_data, Tr, Tr_neg, num_users=None, num_items=None):
        if num_users is None:
            self.num_users = np.max(train_data[:, 0])+1
        else:
            self.num_users = num_users
        if num_items is None:
            self.num_items = np.max(train_data[:, 1])+1
        else:
            self.num_items = num_items
        self.users = set(xrange(self.num_users))
        self.items = set(xrange(self.num_items))
        self.U = np.random.rand(self.num_users, self.d)
        self.V = np.random.rand(self.num_items, self.d)
        valid_data = self.sample_validation_data(Tr, Tr_neg)
        num_valid = valid_data.shape[0]
        curr_theta = self.theta
        last_loss = BPR.compute_loss(
            valid_data, self.U, self.V, self.reg_u, self.reg_i,
            num_valid, self.num_users, self.num_items, self.d)
        for e in xrange(self.max_iter):
            tic = time.clock()
            pairwise_data = self.random_sample_triples(train_data, Tr_neg)
            num_pairs = pairwise_data.shape[0]
            ii = np.random.permutation(num_pairs)
            BPR.gradient_update(
                pairwise_data[ii, :], self.U, self.V, curr_theta, self.reg_u,
                self.reg_i, num_pairs, self.d)
            if np.isnan(np.linalg.norm(self.U, 'fro')) or \
               np.isnan(np.linalg.norm(self.V, 'fro')):
                print "early stop"
                break
                # import sys
                # sys.exit()
            curr_loss = BPR.compute_loss(
                valid_data, self.U, self.V, self.reg_u, self.reg_i,
                num_valid, self.num_users, self.num_items,
                self.d)
            delta_loss = (curr_loss-last_loss)/last_loss
            if self.outf > 0:
                print "epoch:%d, CurrLoss:%.6f, DeltaLoss:%.6f, Time:%.6f" % (
                    e, curr_loss, delta_loss, time.clock()-tic)
            if abs(delta_loss) < 1e-5:
                break
            last_loss, curr_theta = curr_loss, 0.9*curr_theta
        print "complete the learning of bprmf model"

    def predict_individual(self, u, inx, k):
        val = np.dot(self.U[u, :], self.V[inx, :].transpose())
        ii = np.argsort(val)[::-1][:k]
        return inx[ii]

    def evaluation(self, Tr_neg, Te, positions=[5, 10, 15]):
        from evaluation import precision
        from evaluation import recall
        from evaluation import nDCG
        prec = np.zeros(len(positions))
        rec = np.zeros(len(positions))
        ndcg = np.zeros(len(positions))
        map_value, auc_value = 0.0, 0.0
        for u in Te:
            val = np.dot(self.U[u, :], self.V.transpose())
            inx = Tr_neg[u]["items"]
            A = set(Te[u])
            B = set(inx) - A
            # compute precision and recall
            ii = np.argsort(val[inx])[::-1][:max(positions)]
            prec += precision(Te[u], inx[ii], positions)
            rec += recall(Te[u], inx[ii], positions)
            # ndcg += nDCGAtN(Te[u], inx[ii], 10)
            ndcg += np.array([nDCG(Te[u], inx[ii], p) for p in positions])
            # compute map and AUC
            pos_inx = np.array(list(A))
            neg_inx = np.array(list(B))
            map_value += BPR.mean_average_precision(pos_inx, neg_inx, val)
            auc_value += BPR.auc_computation(pos_inx, neg_inx, val)
        return map_value, auc_value, ndcg, prec, rec

    def __str__(self):
        return "Recommender: bprmf, num_factors:%s, reg_u:%s, reg_i:%s, theta:%s, max_iter:%s, seed:%s" % (self.d, self.reg_u, self.reg_i, self.theta, self.max_iter, self.seed)


class SOBPR(BPRMF):

    def __init__(self, num_factors=100, reg_u=0.015, reg_i=0.015, tau_u=0.015, tau_i=0.015, theta=0.05,
                 max_iter=100, eta=0.01, flag=1):
        self.d = num_factors
        self.theta = theta
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.tau_u = tau_u
        self.tau_i = tau_i
        self.max_iter = max_iter
        self.eta = eta
        self.flag = flag

    def fix_model(self, train_data, Tr, Tr_neg):
        self.num_users = np.max(train_data[:, 0])+1
        self.num_items = np.max(train_data[:, 1])+1
        self.U = np.random.rand(self.num_users, self.d)
        self.V = np.random.rand(self.num_items, self.d)
        self.rho_U = np.zeros((self.num_users, self.d))
        self.rho_V = np.zeros((self.num_items, self.d))
        self.hU = np.zeros((self.num_users, self.d))
        self.hV = np.zeros((self.num_items, self.d))
        if self.flag == 2:
            self.SigMu = np.ones((self.num_users, self.d))
            self.SigMv = np.ones((self.num_items, self.d))
        valid_data = self.sample_validation_data(Tr, Tr_neg, ratio=0.5)
        BPR.socf_bpr_train(self, valid_data, train_data, Tr, Tr_neg)

    def __str__(self):
        if self.flag == 1:
            return "Recommender: sobpr-firstorder, num_factors:%s, lmbda:%s,%s, tau:%s, %s, theta:%s, max_iter:%s" % (self.d, self.reg_u, self.reg_i, self.tau_u, self.tau_i, self.theta, self.max_iter)
        elif self.flag == 2:
            return "Recommender: sobpr-secondorder, num_factors:%s, lmbda:%s, eta:%s, max_iter:%s" % (self.d, self.reg_u, self.eta, self.max_iter)


def data_process(train_file, test_file):
    import data_io
    train_data = data_io.load_history_array(train_file)
    Te = data_io.load_history_dict(test_file)
    Tr = defaultdict(lambda: defaultdict(int))
    items = []
    for u, i in train_data:
        Tr[u][i] += 1
        items.append(i)
    items = set(items)
    Tr_neg = {}
    for u in Tr:
        x = list(items-set(Tr[u].keys()))
        Tr_neg[u] = {"items": np.array(x), "num": len(x)}
    return train_data, Tr, Tr_neg, Te


if __name__ == '__main__':
    # folder = '../datasets/ml/'
    dataset = 'filmtrust'
    recommender = 'bprmf'
    train_file = "../datasets/ml10m_training_threshold_3.0_percentage_0.8.txt"
    test_file = "../datasets/ml10m_testing_threshold_3.0_percentage_0.8.txt"
    # train_file = folder+dataset+'_training.txt'
    # test_file = folder+dataset+'_testing.txt'
    train_data, Tr, Tr_neg, Te = data_process(train_file, test_file)
    positions = [5, 10, 15]
    optimal_para, optimal_results, max_prec = '', '',  0

    if recommender == 'bprmf':
        inf = open('../output/bprmf_results.txt', 'a+')
        for d in xrange(4):  # 10, 30, 50
            for x in np.arange(-6, -5):  # -6, -1
                for y in np.arange(-6, -5):  # -6, -1
                    cmd_str = 'Dataset:'+dataset+'\n'
                    d, x, y = 50, -6, -2
                    model = BPRMF(num_factors=d, reg_u=2**(x), reg_i=2**(x), theta=2**(y), max_iter=200, outf=0)
                    cmd_str += str(model)
                    print cmd_str
                    model.fix_model(train_data, Tr, Tr_neg)
                    map_value, auc_value, ndcg, prec, rec = model.evaluation(Tr_neg, Te, positions)
                    # results = 'MAP: %s AUC:%s nDCG:%s ' % (map_value/len(Te.keys()), auc_value/len(Te.keys()), ndcg/len(Te.keys()))
                    results = ' '.join(['P@%d:%.6f' % (positions[i], prec[i]/len(Te.keys())) for i in xrange(len(positions))])+' '
                    results += ' '.join(['R@%d:%.6f' % (positions[i], rec[i]/len(Te.keys())) for i in xrange(len(positions))])+' '
                    results += ' '.join(['NDCG@%d:%.6f' % (positions[i], ndcg[i]/len(Te.keys())) for i in xrange(len(positions))])
                    inf.write(cmd_str+'\n'+results+'\n')
                    print results
                    if prec[0] > max_prec:
                        optimal_para = str(model)
                        optimal_results = results
                        max_prec = prec[0]
        print "\nthe optimal parameters and results are as follows:\n%s\n%s" % (optimal_para, optimal_results)
        inf.close()

    if recommender == 'socf':
        inf = open('../data/socf_results.txt', 'a+')
        for x in np.arange(-6, 0):
            for y in np.arange(-6, 0):
                for z in np.arange(-6, -5):
                    cmd_str = 'Dataset:'+dataset+'\n'
                    model = SOBPR(num_factors=10, reg_u=10**(x), eta=2**(y), max_iter=100, flag=2)
                    cmd_str += str(model)
                    print cmd_str
                    model.fix_model(train_data, Tr, Tr_neg)
                    x1, x2 = float(np.count_nonzero(model.U))/model.U.size, float(np.count_nonzero(model.V))/model.V.size
                    spr = 'U density:%.6f, V density:%.6f' % (x1, x2)
                    map_value, auc_value, ndcg, prec, rec = model.evaluation(Tr_neg, Te, positions)
                    results = 'MAP: %s AUC:%s nDCG:%s ' % (map_value/len(Te.keys()), auc_value/len(Te.keys()), ndcg/len(Te.keys()))
                    results += ' '.join(['P@%d:%.6f' % (positions[i], prec[i]/len(Te.keys())) for i in xrange(len(positions))])+' '
                    results += ' '.join(['R@%d:%.6f' % (positions[i], rec[i]/len(Te.keys())) for i in xrange(len(positions))])
                    inf.write(cmd_str+'\n'+spr+'\n'+results+'\n')
                    print spr+'\n'+results
                    if ndcg > max_ndcg:
                        optimal_para = str(model)
                        optimal_results = spr+'\n'+results
                        max_ndcg = ndcg
        print "\nthe optimal parameters and results are as follows:\n%s\n%s" % (optimal_para, optimal_results)
        inf.close()
