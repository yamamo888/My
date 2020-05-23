# -*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import cycle


class NankaiData:
    def __init__(self):
        
        self.logsfullpath1 = glob.glob(os.path.join('logs', 'b2b3b4b5b6', '*txt'))
        self.logsfullpath2 = glob.glob(os.path.join('logs', 'tmp300', '*txt'))
        
        # init batch count
        self.batchCnt1 = 0
        self.batchCnt2 = 0
        
        # size of data
        self.nTrain1 = len(self.logsfullpath1)
        self.nTrain2 = len(self.logsfullpath2)
        
    # ----
    def LSTM(self, x, seq, reuse=False):

        nHidden=128
        
        with tf.variable_scope("LSTM") as scope:
            if reuse:
                scope.reuse_variables()
            
            # multi cell
            cells = []
            # 1st LSTM
            cell1 = tf.contrib.rnn.LSTMCell(nHidden, use_peepholes=True)
            # 2nd LSTM
            cell2 = tf.contrib.rnn.LSTMCell(nHidden, use_peepholes=True)
        
            cells.append(cell1)
            cells.append(cell2)
            
            cell = tf.nn.rnn_cell.MultiRNNCell(cells)
            
            outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=seq)
            
            # outputs [None,None,HIDDEN] 
            # states[-1] tuple (Ct [None,128], Ht [None,128])
            return outputs, states[-1]
    # ----
    
    # ----
    def EvalData(self):
        
        fID = 190
       
        # eq.year ----        
        rirekipath = os.path.join('features','eval','nankairireki.pkl')
        with open(rirekipath ,'rb') as fp:
            data = pickle.load(fp)
        xrireki = data[fID,:,:]
        self.yEval = [np.where(xrireki[:,0]>0)[0], np.where(xrireki[:,1]>0)[0], np.where(xrireki[:,2]>0)[0]] # [[eq.year in nk], [eq.year in tnk], [eq.year in tk]]
        
        pdb.set_trace()
        
        # interval
        nk = self.yEval[0][1:] - self.yEval[0][:-1]
        tnk = self.yEval[1][1:] - self.yEval[1][:-1]
        tk = self.yEval[2][1:] - self.yEval[2][:-1]
        # evaluation input (5cell)
        xEval = np.concatenate([nk,nk,tnk,tnk,tk],0)
        # length of interval
        seqEval = np.max([len(nk),len(tnk),len(tk)])
        
        return xEval, self.yEval, seqEval
    # ----
           
    # ----
    def nextBatch(self, nBatch=100):
        '''
        batchX: eq.intervals
        batchY: eq.year
        '''
        
        # index
        sInd1 = nBatch * self.batchCnt1
        eInd1 = sInd1 + nBatch
        
        sInd2 = nBatch * self.batchCnt2
        eInd2 = sInd2 + nBatch
        pdb.set_trace()
        # Select path
        batchPath1 = self.logsfullpath1[sInd1:eInd1]
        batchPath2 = self.logsfullpath2[sInd2:eInd2]
        batchPath = batchPath1.append(batchPath2)
        
        # Loading logs
        makeInterval = cycle.Cycle()
        
        interval_list = []
        flag = False
        for path in batchPath:
            makeInterval.loadBV(path)
            makeInterval.convV2YearlyData()
            
            # output dataset, years:list
            years, _ = makeInterval.calcYearMSE(self.yEval)
            # input dataset, intervals:list
            intervals, seq = makeInterval.calcInterval()
            
            interval_list = interval_list.append(intervals)
            
            if not flag:
                bathcSeq = seq
                batchY1 = years[0][:9]
                batchY2 = years[1][:9]
                batchY3 = years[2][:7]
                flag = True
            else:
                batchSeq = np.hstack([batchSeq, seq])
                batchY1 = np.vstack([batchY1, years[0][:9]])
                batchY2 = np.vstack([batchY2, years[0][:9]])
                batchY3 = np.vstack([batchY3, years[0][:7]])
                
        batchY = [batchY1, batchY2, batchY3]
        
        # stand zero-padding length
        maxseq = np.max(batchSeq)
        
        flag = True
        for intervals in interval_list:
            # zero-padding length(maxseq)
            interval_nk1 = np.pad(intervals[0], [0, maxseq-len(intervals[0])], 'constant')
            interval_nk2 = np.pad(intervals[1], [0, maxseq-len(intervals[1])], 'constant')
            interval_tnk1 = np.pad(intervals[2], [0, maxseq-len(intervals[2])], 'constant')
            interval_tnk2 = np.pad(intervals[3], [0, maxseq-len(intervals[3])], 'constant')
            interval_tk = np.pad(intervals[4], [0, maxseq-len(intervals[4])], 'constant')
            
            intervals = np.concatenate([interval_nk1,interval_nk2,interval_tnk1,interval_tnk2,interval_tk],0)
            
            if not flag:
                batchX = intervals
                flag = True
            else:
                batchX = np.vstack([batchX, intervals])
                
        if eInd1 + nBatch > self.nTrain1:
            self.batchCnt1 = 0
        else:
            self.batchCnt1 += 1
        
        if eInd2 + nBatch > self.nTrain2:
            self.batchCnt2 = 0
        else:
            self.batchCnt2 += 1
    
    return batchX, batchY, batchSeq
    # ----
   
        
    
    