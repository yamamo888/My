# -*- coding: utf-8 -*-

import os
import glob
import pdb
import pickle

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
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
        '''
        # Loading logs
        makeInterval = cycle.Cycle()
        
        self.EvalData()       
        
        #filename = ['b2b3b4b5b6205-300','tmp300','b2b3b4b5b60-100','b2b3b4b5b6105-200','b2b3b4b5b6400-450'] 
        
        flag = False
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

            with open(os.path.join(self.featurePath,'interval',f'intervalSeqXY_{fID}.pkl'),'wb') as fp:
                pickle.dump(seqs, fp)
                pickle.dump(Intervals, fp)
                pickle.dump(Years, fp)
                pickle.dump(Bs, fp)'''
            
        with open(os.path.join(self.featurePath,'interval',f'intervalSeqXY_tmp300_slip1.pkl'),'rb') as fp:
            Seqs = pickle.load(fp)
            Intervals = pickle.load(fp)
            Years = pickle.load(fp)
            Paramb = pickle.load(fp)
        pdb.set_trace()
        
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
    # ----
    
    # ----
    def makeNearYearData(self):
        
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_slip1.pkl'),'rb') as fp:
            Seqs = pickle.load(fp)
            Intervals = pickle.load(fp)
            Years = pickle.load(fp)
            Paramb = pickle.load(fp)
            
        
        self.IntervalEvalData()
        gt_year_nk,gt_year_tnk,gt_year_tk = self.yEval[0],self.yEval[1],self.yEval[2]
        cnt = 0
        Flag = False
        for years in Years:
            pdb.set_trace()
            print(cnt)
            cnt += 1
            year_nk,year_tnk,year_tk = np.trim_zeros(years[:,0]),np.trim_zeros(years[:,1]),np.trim_zeros(years[:,2])
            flag = False
            for pyear,gyear in zip([year_nk,year_tnk,year_tk],[gt_year_nk,gt_year_tnk,gt_year_tk]):
                
                # exact year > simulated year
                if len(pyear) <= len(gyear):
                    
                    near_pyear = pyear
                
                else:
                    # exact year == simulated year
                    gyears = gyear.repeat(pyear.shape[0],0).reshape(-1,pyear.shape[0])
                    pyears = pyear.repeat(gyear.shape[0],0).reshape(-1,gyear.shape[0])
                    # minimum year index & year 
                    inds = [np.argmin(np.abs(g-p)) for g,p in zip(gyears,pyears.T)]
                    near_pyear = [pyear[ind] for ind in inds]
                    # del multiple
                    near_pyear = np.unique(np.array(near_pyear))
                
                #pdb.set_trace()
                near_pintervals = (near_pyear[1:]-near_pyear[:-1])
                near_pinterval = near_pintervals[near_pintervals>0]
            
                # zero-padding
                zero_year = np.pad(near_pyear, [0,9-len(near_pyear)], 'constant')
                zero_interval = np.pad(near_pinterval, [0,8-len(near_pinterval)], 'constant')
                
                if not flag:
                    zero_years = zero_year
                    zero_intervals = zero_interval
                    flag = True
                else:
                    zero_years = np.vstack([zero_years,zero_year])
                    zero_intervals = np.vstack([zero_intervals,zero_interval])
                    
            if not Flag:
                zeroYears = zero_years.T[np.newaxis]
                zeroIntervals = zero_intervals.T[np.newaxis]
                Flag = True
            else:
                # [data,9(zero-padding),3(cell)],[data,8(zero-padding),3(cell)] 
                zeroYears = np.vstack([zeroYears,zero_years.T[np.newaxis]])
                zeroIntervals = np.vstack([zeroIntervals,zero_intervals.T[np.newaxis]])
            #pdb.set_trace()
       
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_near_back0.pkl'),'wb') as fp:
            pickle.dump(Seqs, fp)
            pickle.dump(zeroIntervals, fp)
            pickle.dump(zeroYears, fp)
            pickle.dump(Paramb, fp)
    # ----
    
    # ----
    def loadIntervalTrainTestData(self):
        
        with open(os.path.join(self.featurePath,'interval',f'lstm.pkl'),'rb') as fp:
            self.seqTrain = pickle.load(fp)
            self.intervalTrain = pickle.load(fp)
            self.yearTrain = pickle.load(fp)
            self.parambTrain = pickle.load(fp)
        
        with open(os.path.join(self.featurePath,'interval',f'test_intervalSeqXY_tmp300_near5000_back0.pkl'),'rb') as fp:
            seqTest = pickle.load(fp)
            intervalTest = pickle.load(fp)
            yearTest = pickle.load(fp)
            parambTest = pickle.load(fp)
        
        seqTest = seqTest[:100]
        intervalTest = intervalTest[:100]
        yearTest = yearTest[:100]
        parambTest = parambTest[:100]
        
        return intervalTest, parambTest, yearTest, seqTest
    # ----

    # ----
    def IntervalEvalData(self):
        
        fID = 190
       
        # eq.year ----        
        rirekipath = os.path.join(self.featurePath,'eval','nankairireki.pkl')
        with open(rirekipath ,'rb') as fp:
            data = pickle.load(fp)
        xrireki = data[fID,:,:]
        self.yEval = [np.where(xrireki[:,0]>0)[0], np.where(xrireki[:,1]>0)[0], np.where(xrireki[:,2]>0)[0]] # [[eq.year in nk], [eq.year in tnk], [eq.year in tk]]
    
        '''
        sns.set_style('dark')
        #pdb.set_trace()
        fig, figInds = plt.subplots(nrows=3, sharex=True)
        for figInd in np.arange(len(figInds)):
            figInds[figInd].plot(np.arange(1400), xrireki[:,figInd], color='coral')
        
        plt.savefig("Nankai.png")
        plt.close()
        '''
       
        # interval nk:8.tnk:8.tk:6
        nk = self.yEval[0][1:] - self.yEval[0][:-1]
        tnk = self.yEval[1][1:] - self.yEval[1][:-1]
        tk = self.yEval[2][1:] - self.yEval[2][:-1]
        
        # zoro-padding tk 6->8
        tk = np.pad(tk, [0,2], 'constant')

        # evaluation input, [1(data),8(interval),5(cell)]
        #xEval = np.concatenate([nk[:,np.newaxis],nk[:,np.newaxis],tnk[:,np.newaxis],tnk[:,np.newaxis],tk[:,np.newaxis]],1)[np.newaxis]
        self.xEval = np.concatenate([nk[:,np.newaxis],tnk[:,np.newaxis],tk[:,np.newaxis]],1)[np.newaxis]
        
        # length of interval, array(8)
        seqEval = np.array([np.max([len(nk),len(tnk),len(tk)])])

        return self.xEval, self.yEval, seqEval
    # ----
   
    # ----
    def LSTM(self, x, seq, reuse=False):

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
            
            outputs, states = tf.compat.v1.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32, sequence_length=seq)
            
            # outputs [None,None,HIDDEN] 
            # states[-1] tuple (Ct [None,128], Ht [None,128])
            return outputs, states[-1]
    # ----
    
    # ----
    def nextBatch(self, index):
        '''
        batchX: eq.intervals. [data, max of eq.length, 5(cell)]
        batchY: paramb
        batchCycleY: eq.years
        batchSeq: length of maximum eq.intervals
        '''
              
        batchX = self.intervalTrain[index]
        batchY = self.parambTrain[index]
        batchCycleY = self.yearTrain[index]
        batchSeq = self.seqTrain[index]
        
        batchXY = [batchX, batchY, batchCycleY, batchSeq]
        
        return batchXY
    # ----
   
#NankaiData().makeIntervalData()
#NankaiData().makeNearYearData()
#NankaiData().loadIntervalTrainTestData()