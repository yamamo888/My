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
        
        #self.logsfullpath1 = glob.glob(os.path.join('logs', 'b2b3b4b5b6205-300', '*txt'))
        self.logsfullpath2 = glob.glob(os.path.join('logs', 'tmp300', '*txt'))
        
        # init batch count
        #self.batchCnt1 = 0
        self.batchCnt2 = 0
        
        # size of data
        #self.nTrain1 = len(self.logsfullpath1)
        self.nTrain2 = len(self.logsfullpath2)
        
        self.isLSTM = True
    
    # ----
    def makeIntervalData(self):
        
        # Loading logs
        makeInterval = cycle.Cycle()
        
        self.EvalData()       
        interval_list = []
        flag = False
        #filename = ['b2b3b4b5b6205-300','tmp300','b2b3b4b5b60-100','b2b3b4b5b6105-200','b2b3b4b5b6400-450'] 
        filename = ['b2b3b4b5b6105-200'] 

        flag = False
        #for fID in ['tmp300','b2b3b4b5b6205-300']:
        for fID in filename:
            logspath = glob.glob(os.path.join('logs',f'{fID}','*.txt'))
            cnt = 0
            for logpath in logspath:
                print(f'{fID}:{len(logspath)-cnt}')
                cnt += 1
                B,_ = makeInterval.loadBV(logpath)
                B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
            
                makeInterval.convV2YearlyData(isLSTM=self.isLSTM)
                
                # output dataset, years:list
                years, _ = makeInterval.calcYearMSE(self.yEval,isLSTM=self.isLSTM)
                # zero-padding array[500,]
                years = [np.pad(year, [0, 500-len(year)], 'constant') for year in years] 
                years = np.concatenate([years[0][:,np.newaxis],years[1][:,np.newaxis],years[2][:,np.newaxis]],1)
                    
                # input dataset, intervals:list
                intervals, seq = makeInterval.calcInterval()
                # list[5]
                intervals = [np.pad(interval, [0, 500-len(interval)], 'constant') for interval in intervals] 
                intervals = np.concatenate([intervals[0][:,np.newaxis],intervals[1][:,np.newaxis],intervals[2][:,np.newaxis],intervals[3][:,np.newaxis],intervals[4][:,np.newaxis]],1)

                if not flag:
                    seqs = np.array([seq])
                    Intervals = intervals[np.newaxis]
                    Years = years[np.newaxis]
                    Bs = B[np.newaxis]
                    flag = True
                else:
                    seqs = np.hstack([seqs, np.array([seq])])
                    Intervals = np.vstack([Intervals, intervals[np.newaxis]])
                    Years = np.vstack([Years, years[np.newaxis]])
                    Bs = np.vstack([Bs, B[np.newaxis]])

            with open(os.path.join('features','interval','intervalSeqXY_{fID}.pkl'),'wb') as fp:
                pickle.dump(seqs, fp)
                pickle.dump(Intervals, fp)
                pickle.dump(Years, fp)
                pickle.dump(Bs, fp)
    # ----    

    # ----
    def LSTM(self, x, seq, reuse=False):

        nHidden=128
        
        with tf.compat.v1.variable_scope("LSTM") as scope:
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
    
        # interval nk:8.tnk:8.tk:6
        nk = self.yEval[0][1:] - self.yEval[0][:-1]
        tnk = self.yEval[1][1:] - self.yEval[1][:-1]
        tk = self.yEval[2][1:] - self.yEval[2][:-1]
        
        # zoro-padding tk 6->8
        tk = np.pad(tk, [0,2], 'constant')

        # evaluation input, [1(data),8(interval),5(cell)]
        xEval = np.concatenate([nk[:,np.newaxis],nk[:,np.newaxis],tnk[:,np.newaxis],tnk[:,np.newaxis],tk[:,np.newaxis]],1)[np.newaxis]
        # length of interval, array(8)
        seqEval = np.array([np.max([len(nk),len(tnk),len(tk)])])

        return xEval, self.yEval, seqEval
    # ----
           
    # ----
    def nextBatch(self, nBatch=100):
        '''
        batchX: eq.intervals. [data, max of eq.length, 5(cell)]
        batchY: eq.year. list
        batchSeq: length of maximum eq.intervals
        '''
        
        # index
        #sInd1 = nBatch * self.batchCnt1
        #eInd1 = sInd1 + nBatch
        
        sInd2 = nBatch * self.batchCnt2
        eInd2 = sInd2 + nBatch
        
        # Select path
        #batchPath1 = self.logsfullpath1[sInd1:eInd1]
        batchPath2 = self.logsfullpath2[sInd2:eInd2]
        #batchPath = []
        #batchPath.append(batchPath1)
        #batchPath.append(batchPath2)
        # flatten [2] -> [batchSize*2]
        #batchPath = sum(batchPath, [])
        
        # Loading logs
        makeInterval = cycle.Cycle()
        
        interval_list = []
        flag = False
        #for path in batchPath:
        for path in batchPath2:
            
            makeInterval.loadBV(path)
            makeInterval.convV2YearlyData(isLSTM=self.isLSTM)
            
            # eq.year list[nk1,nk2,tnk1,tnk2,tk]
            years, _ =  makeInterval.calcYearMSE(self.yEval,isLSTM=self.isLSTM)
            
            # if years < len([8,8,6])
            if len(years[0]) < 8:
                years[0] = np.pad(years[0], [0,8-len(years[0])], 'constant')
            if len(years[1]) < 8:
                years[1] = np.pad(years[1], [0,8-len(years[1])], 'constant')
            if len(years[2]) < 6:
                years[2] = np.pad(years[2], [0,6-len(years[2])], 'constant')

            # intervals:list, seq: max length of year
            intervals, seq = makeInterval.calcInterval()
            
            interval_list.append(intervals)
            
            if not flag:
                batchSeq = np.array([seq])
                # nk
                batchY1 = years[0][:8]
                # tnk
                batchY2 = years[1][:8]
                # tk
                batchY3 = years[2][:6]
                flag = True
            else:
                batchSeq = np.hstack([batchSeq, np.array([seq])])

                batchY1 = np.vstack([batchY1, years[0][:8]])
                batchY2 = np.vstack([batchY2, years[1][:8]])
                batchY3 = np.vstack([batchY3, years[2][:6]])
        
        batchY = [batchY1, batchY2, batchY3]
        
        # stand zero-padding length
        maxseq = np.max(batchSeq)
        
        # zero-padding size(maxseq)
        flag = False
        for intervals in interval_list:
            # zero-padding length(maxseq)
            interval_nk1 = np.pad(intervals[0], [0, maxseq-len(intervals[0])], 'constant')
            interval_nk2 = np.pad(intervals[1], [0, maxseq-len(intervals[1])], 'constant')
            interval_tnk1 = np.pad(intervals[2], [0, maxseq-len(intervals[2])], 'constant')
            interval_tnk2 = np.pad(intervals[3], [0, maxseq-len(intervals[3])], 'constant')
            interval_tk = np.pad(intervals[4], [0, maxseq-len(intervals[4])], 'constant')
            
            intervals = np.concatenate([interval_nk1[:,np.newaxis],interval_nk2[:,np.newaxis],interval_tnk1[:,np.newaxis],interval_tnk2[:,np.newaxis],interval_tk[:,np.newaxis]],1)
            
            if not flag:
                batchX = intervals[np.newaxis]
                flag = True
            else:
                batchX = np.vstack([batchX, intervals[np.newaxis]])
                
        #if eInd1 + nBatch > self.nTrain1:
            #self.batchCnt1 = 0
        #else:
            #self.batchCnt1 += 1
        
        if eInd2 + nBatch > self.nTrain2:
            self.batchCnt2 = 0
        else:
            self.batchCnt2 += 1
    
        return batchX, batchY, batchSeq
    # ----
   
NankaiData().makeIntervalData()       
    
    
