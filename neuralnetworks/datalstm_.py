# -*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

import cycle


class NankaiData:
    def __init__(self, isLSTM=True):
        
        # path ----
        self.featurePath = 'features'
    
    # ----
    def makeIntervalData(self):
        
        # Loading logs
        makeInterval = cycle.Cycle()
        
        #self.EvalData()       
        
        #filename = ['b2b3b4b5b6205-300','b2b3b4b5b60-100','b2b3b4b5b6105-200','b2b3b4b5b6400-450'] 
        filename = ['tmp300'] 
        
        flag = False
        for fID in filename:
            logspath = glob.glob(os.path.join('logs',f'{fID}','*.txt'))
            cnt = 0
            for logpath in logspath:
                print(f'{fID}:{len(logspath)-cnt}')
                cnt += 1
                
                B,_ = makeInterval.loadBV(logpath)
                B = np.concatenate([B[2,np.newaxis],B[4,np.newaxis],B[5,np.newaxis]],0)
                allyears, onehotYear = makeInterval.convV2YearlyData(isZeroYear=True)
                
                # zero-padding array[500,]
                years = [np.pad(year, [0, 200-len(year)], 'constant') for year in [allyears[1],allyears[3],allyears[4]]] 
                years = np.concatenate([years[0][:,np.newaxis],years[1][:,np.newaxis],years[2][:,np.newaxis]],1)
                    
                # input dataset, intervals:list
                intervals, seq = makeInterval.calcInterval(allyears)
                # list[5]
                intervals = [np.pad(interval, [0, 200-len(interval)], 'constant') for interval in intervals] 
                intervals = np.concatenate([intervals[0][:,np.newaxis],intervals[1][:,np.newaxis],intervals[2][:,np.newaxis],intervals[3][:,np.newaxis],intervals[4][:,np.newaxis]],1)
                if not flag:
                    seqs = np.array([seq])
                    Intervals = intervals[np.newaxis]
                    Years = years[np.newaxis]
                    onehotYears = onehotYear[np.newaxis]
                    Bs = B[np.newaxis]
                    flag = True
                else:
                    seqs = np.hstack([seqs, np.array([seq])])
                    Intervals = np.vstack([Intervals, intervals[np.newaxis]])
                    Years = np.vstack([Years, years[np.newaxis]])
                    onehotYears = np.vstack([onehotYears, onehotYear[np.newaxis]])
                    Bs = np.vstack([Bs, B[np.newaxis]])

            with open(os.path.join(self.featurePath,'interval',f'intervalSeqXYonehotY_{fID}.pkl'),'wb') as fp:
            #with open(os.path.join(self.featurePath,'interval',f'practice.pkl'),'wb') as fp:
                pickle.dump(seqs, fp, protocol=4)
                pickle.dump(Intervals, fp, protocol=4)
                pickle.dump(Years, fp, protocol=4)
                pickle.dump(onehotYears, fp, protocol=4)
                pickle.dump(Bs, fp, protocol=4)
                
        '''    
        with open(os.path.join(self.featurePath,'interval',f'intervalSeqXY_tmp300_slip1.pkl'),'rb') as fp:
            Seqs = pickle.load(fp)
            Intervals = pickle.load(fp)
            Years = pickle.load(fp)
            Paramb = pickle.load(fp)
        #pdb.set_trace()
        
        #Intervals = intervals[:,:8,:]
        #Years = years[:,:8,:]
        
        nData = Intervals.shape[0]
        nTrain = int(nData * 0.8)
        randInd = np.random.permutation(nData)
        
        # Separate train & test
        seqTrain = Seqs[randInd[:nTrain]]
        intervalTrain = Intervals[randInd[:nTrain]]
        yearTrain = Years[randInd[:nTrain]]
        parambTrain = Paramb[randInd[:nTrain]]
        
        seqTest = Seqs[randInd[nTrain:]]
        intervalTest = Intervals[randInd[nTrain:]]
        yearTest = Years[randInd[nTrain:]]
        parambTest = Paramb[randInd[nTrain:]]
        
        with open(os.path.join(self.featurePath,'interval',f'train_intervalSeqXY_tmp300_slip1.pkl'),'wb') as fp:
            pickle.dump(seqTrain, fp)
            pickle.dump(intervalTrain, fp)
            pickle.dump(yearTrain, fp)
            pickle.dump(parambTrain, fp)
    
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_slip1.pkl'),'wb') as fp:
            pickle.dump(seqTest, fp)
            pickle.dump(intervalTest, fp)
            pickle.dump(yearTest, fp)
            pickle.dump(parambTest, fp)
        '''
    # ----
    
    # ----
    def loadIntervalTrainTestData(self):
        
        with open(os.path.join(self.featurePath,'interval',f'train_intervalSeqXY_tmp300_near5000_back0.pkl'),'rb') as fp:
            self.seqTrain = pickle.load(fp)
            self.intervalTrain = pickle.load(fp)
            self.yearTrain = pickle.load(fp)
            self.parambTrain = pickle.load(fp)
        #pdb.set_trace()
        with open(os.path.join(self.featurePath,'interval',f'train_intervalSeqXY_tmp300_near_back0_onehotyear.pkl'),'rb') as fp:
            self.onehot_yearTrain = pickle.load(fp)
        
        '''
        onehot_yearTrain = np.zeros([self.yearTrain.shape[0],1400,3])
        
        for i in np.arange(self.yearTrain.shape[0]):
            
            print(self.yearTrain.shape[0]-i)
            
            ind_nk = np.trim_zeros(self.yearTrain[i][:,0])
            ind_tnk = np.trim_zeros(self.yearTrain[i][:,1])
            ind_tk = np.trim_zeros(self.yearTrain[i][:,2])
            
            onehot_yearTrain[i,ind_nk,0] = 1
            onehot_yearTrain[i,ind_tnk,1] = 1
            onehot_yearTrain[i,ind_tk,2] = 1
        '''
        #with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_slip1.pkl'),'rb') as fp:
        
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_near5000_back0.pkl'),'rb') as fp:
            seqTest = pickle.load(fp)
            intervalTest = pickle.load(fp)
            yearTest = pickle.load(fp)
            parambTest = pickle.load(fp)
        #pdb.set_trace()
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_near_back0_onehotyear.pkl'),'rb') as fp:
            onehot_yearTest = pickle.load(fp)
       
        seqTest = seqTest[:100]
        intervalTest = intervalTest[:100]
        yearTest = yearTest[:100]
        parambTest = parambTest[:100]
        #onehot_yearTest[:100]
        
        return intervalTest, parambTest, yearTest, seqTest
    # ----

    # ----
    def IntervalEvalData(self, cNNRestore=False):
    
        # eq.year ----        
        rirekipath = os.path.join(self.featurePath,'eval','nankairireki.pkl')
        with open(rirekipath ,'rb') as fp:
            data = pickle.load(fp)
        
        #pdb.set_trace()
        
        # all rireki
        if cNNRestore:
            flag = False
            for xdata in data:
                # year
                ynk = np.where(xdata[:,0]>0)[0]
                ytnk = np.where(xdata[:,1]>0)[0]
                ytk = np.where(xdata[:,2]>0)[0]
                # interval
                xnk = ynk[1:] - ynk[:-1]
                xtnk = ytnk[1:] - ytnk[:-1]
                xtk = ytk[1:] - ytk[:-1]
                
                seq = np.array([np.max([len(xnk),len(xtnk),len(xtk)])])
                
                # zero-padding
                ynk = np.pad(ynk, [0,9-len(ynk)], 'constant')
                ytnk = np.pad(ytnk, [0,9-len(ytnk)], 'constant')
                ytk = np.pad(ytk, [0,7-len(ytk)], 'constant')
                
                xnk = np.pad(xnk, [0,8-len(xnk)], 'constant')
                xtnk = np.pad(xtnk, [0,8-len(xtnk)], 'constant')
                xtk = np.pad(xtk, [0,8-len(xtk)], 'constant')
                
                x = np.concatenate([xnk[:,np.newaxis],xtnk[:,np.newaxis],xtk[:,np.newaxis]],1)[np.newaxis]
                        
                if not flag:
                    xEval = x
                    yEvalnk = ynk
                    yEvaltnk = ytnk
                    yEvaltk = ytk
                    seqEval = seq
                    
                    flag = True
                else:
                    xEval = np.vstack([xEval,x])
                    yEvalnk = np.vstack([yEvalnk,ynk])
                    yEvaltnk = np.vstack([yEvaltnk,ytnk])
                    yEvaltk = np.vstack([yEvaltk,ytk])
                    seqEval = np.hstack([seqEval,seq])
                    
            yEval = [yEvalnk, yEvaltnk, yEvaltk]
        
        # only No.190 rireki
        else:    
            fID = 190
       
            xrireki = data[fID,:,:]
            yEval = [np.where(xrireki[:,0]>0)[0], np.where(xrireki[:,1]>0)[0], np.where(xrireki[:,2]>0)[0]] # [[eq.year in nk], [eq.year in tnk], [eq.year in tk]]
    
            # interval nk:8.tnk:8.tk:6
            nk = yEval[0][1:] - yEval[0][:-1]
            tnk = yEval[1][1:] - yEval[1][:-1]
            tk = yEval[2][1:] - yEval[2][:-1]
            
            # zoro-padding tk 6->8
            tk = np.pad(tk, [0,2], 'constant')
    
            # evaluation input, [1(data),8(interval),5(cell)]
            #xEval = np.concatenate([nk[:,np.newaxis],nk[:,np.newaxis],tnk[:,np.newaxis],tnk[:,np.newaxis],tk[:,np.newaxis]],1)[np.newaxis]
            xEval = np.concatenate([nk[:,np.newaxis],tnk[:,np.newaxis],tk[:,np.newaxis]],1)[np.newaxis]
            
            # length of interval, array(8)
            seqEval = np.array([np.max([len(nk),len(tnk),len(tk)])])
            
        return xEval, yEval, seqEval
    # ----
   
    # ----
    def LSTM(self, x, seq, reuse=False):

        #nHidden=120
        nHidden=32
        
        with tf.compat.v1.variable_scope("LSTM") as scope:
            if reuse:
                scope.reuse_variables()
            
            # multi cell
            cells = []
            # 1st LSTM
            cell1 = tf.compat.v1.nn.rnn_cell.LSTMCell(nHidden, use_peepholes=True)
            # 2nd LSTM
            cell2 = tf.compat.v1.nn.rnn_cell.LSTMCell(nHidden, use_peepholes=True)
        
            cells.append(cell1)
            cells.append(cell2)
            
            cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
            
            output, states = tf.compat.v1.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=seq)
            #pdb.set_trace()
            # outputs [None,None,HIDDEN] 
            # states[-1] tuple (Ct [None,128], Ht [None,128])
            return states[-1]
    # ----
    
    # ----
    def nextBatch(self, index):
        '''
        batchX: eq.intervals. [data, max of eq.length, 5(cell)]
        batchY: paramb
        batchCycleY: eq.years
        batchSeq: length of maximum eq.intervals
        '''
        #pdb.set_trace()
        batchX = self.intervalTrain[index]
        batchY = self.parambTrain[index]
        batchCycleY = self.yearTrain[index]
        batchSeq = self.seqTrain[index]
        batchYear = self.onehot_yearTrain[index]
        
        batchXY = [batchX, batchY, batchCycleY, batchSeq, batchYear]
        
        return batchXY
    # ----
   
NankaiData().makeIntervalData()
#NankaiData().makeNearYearData()
#NankaiData().loadIntervalTrainTestData()
