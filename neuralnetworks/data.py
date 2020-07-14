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
    def __init__(self):
        
        # path ----
        self.featurePath = 'features'
    
    # ----
    def makeIntervalData(self):
        
        # Loading logs
        makeInterval = cycle.Cycle()
        
        
        #filename = ['b2b3b4b5b6205-300','tmp300','b2b3b4b5b60-100','b2b3b4b5b6105-200','b2b3b4b5b6400-450'] 
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
                intervals, seq = makeInterval.calcInterval(years)
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
    def makeOnehotYear(self, data, nYear=8000):
        '''
        data: year [data, 150(8), 3]
        nYear: onehot year (train&test->8000, eval->1400)
        '''
        
        onehotYear = np.zeros([data.shape[0],nYear,3])
        
        for i in np.arange(data.shape[0]):
            
            # eq. year (data of zeros-padding, del zero)
            ind_nk = np.trim_zeros(data[i][:,0])
            ind_tnk = np.trim_zeros(data[i][:,1])
            ind_tk = np.trim_zeros(data[i][:,2])
            
            # eq. == 1
            onehotYear[i,ind_nk,0] = 1
            onehotYear[i,ind_tnk,1] = 1
            onehotYear[i,ind_tk,2] = 1
                
        return onehotYear 
    # ----
    
    # ----
    def TrainTest(self):
        '''
        seq: for RNN
        interval: input RNN
        year: onehot for odeNN
        paramb: output paramNN
        '''
        
        with open(os.path.join(self.featurePath,'interval',f'train_intervalSeqXY.pkl'),'rb') as fp:
            self.seqTrain = pickle.load(fp)
            self.intervalTrain = pickle.load(fp)
            yearTrain = pickle.load(fp)
            self.parambTrain = pickle.load(fp)
         
        self.onehotyearTrain = self.makeOnehotYear(yearTrain)    
        
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY.pkl'),'rb') as fp:
            seqTest = pickle.load(fp)
            intervalTest = pickle.load(fp)
            yearTest = pickle.load(fp)
            parambTest = pickle.load(fp)
       
        yearTest = yearTest[:self.nTest]
        onehotyearTest = self.makeOnehotYear(yearTest)
        
        seqTest = seqTest[:self.nTest]
        intervalTest = intervalTest[:self.nTest]
        parambTest = parambTest[:self.nTest]
        
        return intervalTest, seqTest, onehotyearTest, parambTest 
    # ----

    # ----
    def Eval(self, allRireki=False):
    
        # eq.year ----        
        rirekipath = os.path.join(self.featurePath,'eval','nankairireki.pkl')
        with open(rirekipath ,'rb') as fp:
            data = pickle.load(fp)
        
        # all rireki
        if allRireki:
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
            # year
            yEval = [np.where(xrireki[:,0]>0)[0], np.where(xrireki[:,1]>0)[0], np.where(xrireki[:,2]>0)[0]] # [[eq.year in nk], [eq.year in tnk], [eq.year in tk]]
            
            # onehot year
            yonehotEval = self.makeOnehotYear(yEval,nYear=1400)
    
            # interval nk:8.tnk:8.tk:6
            nk = yEval[0][1:] - yEval[0][:-1]
            tnk = yEval[1][1:] - yEval[1][:-1]
            tk = yEval[2][1:] - yEval[2][:-1]
            
            # zoro-padding tk 6->8
            tk = np.pad(tk, [0,2], 'constant')
    
            # evaluation input, [1(data),8(interval),3(cell)]
            xEval = np.concatenate([nk[:,np.newaxis],tnk[:,np.newaxis],tk[:,np.newaxis]],1)[np.newaxis]
            
            # length of interval, array(8)
            seqEval = np.array([np.max([len(nk),len(tnk),len(tk)])])
            
        return xEval, seqEval, yEval, yonehotEval
    # ----
   
    # ----
    def FeatureVec(self, x, seq, reuse=False):

        nHidden = 64
        
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
            
            _, states = tf.compat.v1.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=seq)
            #pdb.set_trace()
            # outputs [None,None,HIDDEN] 
            # states[-1] tuple (Ct [None,128], Ht [None,128])
            return states[-1][1]
    # ----
    
    # ----
    def nextBatch(self, index):
        '''
        batchX: eq.intervals. [data, max of eq.length, 5(cell)]
        batchY: paramb
        batchSeq: length of maximum eq.intervals
        batchYear: eq.years (onehot)
        '''
        #pdb.set_trace()
        batchX = self.intervalTrain[index]
        batchSeq = self.seqTrain[index]
        batchY = self.parambTrain[index]
        batchYear = self.onehotyearTrain[index]
        
        batchXY = [batchX, batchSeq, batchY, batchYear]
        
        return batchXY
    # ----
   
#NankaiData().makeIntervalData()
#NankaiData().makeNearYearData()
#NankaiData().loadIntervalTrainTestData()