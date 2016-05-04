
# coding: utf-8

# In[309]:

import pandas as pd
import scipy as sp
import numpy as np
import csv
import sys
import os
import math
import random as rd

def GibbsSampler(widf, docs,wd,T,ITER, lidf,lwd):

  # In[404]:

  D = len(docs)
  V = max(wd)
  ALPHA = 50/float(T)
  BETA = 5/float(V)
  WBETA = V*BETA
  TALPHA = T*ALPHA

  # In[406]:

  wtp = np.zeros((T,V,V))
  totz = np.zeros((T,1),dtype=int)
  zd = np.zeros((D,1),dtype=int)

  totprob = 0
  maxprob = 0
  prob = np.zeros((T,1))
  di = 0
  for d in docs:
      k = rd.randint(0,T-1)
      zd[di] = k
      totz[k] +=1
      wtp[k,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]+=1
      di+=1


  # In[408]:

  #prob = np.zeros((T,1))
  for itr in range(ITER):
    print 'iter:',itr
    di = 0
    for d in docs:
      u = zd[di]
      totz[u]-=1

      wtp[u,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]-=1
      totprob = 0
      prob = totz + ALPHA
      
      tmp = (wtp[:,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1] + BETA)/(np.sum(wtp[:,wd[d[0]:d[1]]-1,:],min(2,d[1]-d[0]))+WBETA)

      if d[1]-d[0] > 1:
        prob = (prob.transpose() * np.prod(tmp,1))[0]
      else:
        prob = (prob.transpose() * tmp)[0]

      totprob = sum(prob)
      #sample from topic distribution
      r = rd.random() * totprob
      maxprob = prob[0]
      k=0
      while(r > maxprob):
        #print k,r,maxprob
        k+=1
        maxprob += prob[k]
      
      zd[di] = k
      wtp[k,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]+=1
      totz[k]+=1
      di+=1
  return(wtp,[], zd,totz)






def predictDoc(wtp,wd,docs):

  #print wtp
  #wd = rdf.word
  zd = np.zeros(len(docs))
  di = 0
  prob = np.zeros((len(docs),wtp.shape[0]))
  for d in docs:
    tmp1 = (wtp[:,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1])
    if d[1]-d[0] > 1:
      prob[di,:] += np.sum(np.log(tmp1),1)
    else:
      prob[di,:] += np.log(tmp1)
    zd[di]=np.argmax(prob[di,:])
    di+=1
  return (zd,prob)

# In[ ]:




# In[ ]:



