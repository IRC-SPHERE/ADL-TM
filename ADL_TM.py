
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


def GibbsSampler(widf,docs,wd,T,ITER,lidf,lwd,zdin=[]):

  # In[404]:

  D = len(docs)
  V = max(wd)
  M = max(lwd)
  ALPHA = 50/float(T)
  BETA = 5/float(V)
  MU = 5/float(M)
  WBETA = V*BETA
  TALPHA = T*ALPHA
  MMU = M * MU

  # In[406]:

  wtp = np.zeros((T,V,V))
  lwtp = np.zeros((T,M))
  totz = np.zeros((T,1),dtype=int)
  zd = np.zeros((D,1),dtype=int)

  # initialise
  di = 0
  if (len(zdin)>0):
    for d in docs:
      k = zdin[di]
      zd[di] = k
      totz[k] +=1
      wtp[k,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]+=1
      for i in range(d[0]+1,d[1]+1):
        lwtp[k,lwd[i]-1]+= 1
      di+=1
  else:    
    for d in docs:
      k = rd.randint(0,T-1)
      zd[di] = k
      totz[k] +=1
      wtp[k,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]+=1
      #lwtp[k,lwd[d[0]:d[1]+1]-1]+=1
      for i in range(d[0]+1,d[1]+1):
        lwtp[k,lwd[i]-1]+= 1
      di+=1


  # In[408]:

  for itr in range(ITER):
    print 'iter:',itr
    di = 0
    for d in docs:
      #print d
      u = zd[di]
      totz[u]-=1
      #lwtp[u,lwd[d[0]:d[1]]-1]-=1
      for i in range(d[0]+1,d[1]+1):
        lwtp[u,lwd[i]-1]-= 1
      wtp[u,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]-=1
     
      prob = totz + ALPHA
      
      tmp1 = (wtp[:,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1] + BETA)/(np.sum(wtp[:,wd[d[0]:d[1]]-1,:],min(2,d[1]-d[0]))+WBETA)
      
      tmp2 = (lwtp[:,lwd[d[0]:d[1]+1]-1] + MU)/np.repeat(np.array(np.sum(lwtp,1) + MMU,ndmin=2).transpose(), d[1]-d[0]+1, 1)
      if d[1]-d[0] > 1:
        prob = (prob.transpose() * (np.prod(tmp1,1) * np.prod(tmp2,1)))[0]
      else:
        prob = (prob.transpose() * (tmp1 * np.prod(tmp2,1)))[0]
      #print prob
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
      #lwtp[k,lwd[d[0]:d[1]+1]-1]+=1
      for i in range(d[0]+1,d[1]+1):
        lwtp[k,lwd[i]-1]+= 1
      totz[k]+=1
      di+=1
  return(wtp,lwtp, zd, totz)





# In[ ]:

def predictDoc(wtp,lwtp,wd,lwd,docs):
  #print wtp
  zd = np.zeros(len(docs))
  di = 0
  prob = np.zeros((len(docs),wtp.shape[0]))
  for d in docs:
    #print d
    tmp1 = (wtp[:,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1])
    tmp2 = (lwtp[:,lwd[d[0]:d[1]+1]-1])
    if d[1]-d[0] > 1:
      prob[di,:] += (np.sum(np.log(tmp1),1) + np.sum(np.log(tmp2),1))
    elif (d[1]-d[0])==1:
      prob[di,:] += (np.log(tmp1) + np.sum(np.log(tmp2),1))
    else:
      prob[di,:] += np.log(tmp2)
    zd[di]=np.argmax(prob[di,:])
    #print zd[di]
    di+=1
  return (zd,prob)



