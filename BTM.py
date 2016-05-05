
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

def GibbsSampler(widf, docs,wd,T,ITER, lidf=[],lwd=[]):
  
  # In[404]:
  
  D = len(docs)
  V = max(wd)
  W = len(wd)
  
  ALPHA = 50/float(T)
  BETA = 5/float(V)
  WBETA = V*BETA
  TALPHA = T*ALPHA
  
  
  # In[406]:
  
  wtp = np.zeros((T,V,V),dtype=np.float128)
  totz = np.zeros((T,1),dtype=int)
  zw = np.zeros((W,1),dtype=int)
  zd = np.zeros((D,T),dtype=int)
  
  # initialise
  di = 0
  for d in docs:
    k = rd.randint(0,T-1)
    zd[di,k] += 1
    totz[k] += 1
    zw[d[0]] = k
    
    for i in range(d[0]+1,d[1]+1):
      k = rd.randint(0,T-1)
      zd[di,k] += 1
      totz[k] +=1
      zw[i] = k
      wtp[k,wd[i-1]-1,wd[i]-1] += widf[wd[i-1]-1,wd[i]-1]
    
    di+=1
  
  
  # In[408]:
  
  # Gibbs Sampling
  for itr in range(ITER):
    print 'iter:',itr
    di = 0
    for d in docs:
      for i in range(d[0],d[1]+1):
          u = zw[i]
          totz[u]-=1
          zd[di,u]-=1
          tmp1 = 1
          if i > d[0]:
            wtp[u,wd[i-1]-1,wd[i]-1] -= widf[wd[i-1]-1,wd[i]-1]
            tmp1 = (wtp[:,wd[i-1]-1,wd[i]-1] + BETA)/(np.sum(wtp[:,wd[i-1]-1,:],1)+WBETA)
      
          totprob = 0
          prob = (zd[di,:]+ALPHA)/(d[1]-d[0]+1+TALPHA) * tmp1
          
          totprob = sum(prob)
          #sample from topic distribution
          r = rd.random() * totprob
          maxprob = prob[0]
          k=0
          while(r > maxprob):
            #print k,r,maxprob
            k+=1
            maxprob += prob[k]
          
          zw[i] = k
          zd[di,k]+=1
          totz[k]+=1
          if i > d[0]:
            wtp[k,wd[i-1]-1,wd[i]-1] += widf[wd[i-1]-1,wd[i]-1]
      di+=1
  return(wtp, zw, np.argmax(zd,1), totz)





# In[ ]:

def predictDoc(wtp,wd,docs):

  zd = np.zeros(len(docs))
  di = 0
  prob = np.zeros((len(docs),wtp.shape[0]))
  for d in docs:
    tmp1 = wtp[:,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]
    if (d[1]-d[0] > 1):
      prob[di,:] += np.sum(np.log(tmp1),1)
    else :
      prob[di,:] += np.log(tmp1)
               
    zd[di]=np.argmax(prob[di,:])
    di+=1
  return (zd,prob)



