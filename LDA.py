
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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
  
  wtp = np.zeros((T,V),dtype=np.float128)
  totz = np.zeros((T,1),dtype=int)
  zw = np.zeros((W,1),dtype=int)
  zd = np.zeros((D,T),dtype=int)
  
  # initialise
  di = 0
  for d in docs:
    for i in range(d[0],d[1]+1):
      k = rd.randint(0,T-1)
      zd[di,k] += 1
      totz[k] +=1
      zw[i] = k
      wtp[k,wd[i]-1] += widf[wd[i]-1]
    di+=1
  
  
  # In[408]:
  
  # Gibbs Sampling
  for itr in range(ITER):
    print 'iter:',itr
    di = 0
    for d in docs:
      #print 'di is',di, 'd[0] = '+str(d[0]), 'd[1] = '+str(d[1])
      for i in range(d[0],d[1]+1):
          #print 'i = '+str(i)
          u = zw[i]
          totz[u]-=1
          zd[di,u]-=1
          #print 'before',u,zd[di,u]
          wtp[u,wd[i]-1] -= widf[wd[i]-1]
          tmp1 = (wtp[:,wd[i]-1] + BETA)/(np.sum(wtp,1)+WBETA)
          
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
          #print 'after',k, zd[di,k]
          totz[k]+=1
          wtp[k,wd[i]-1] += widf[wd[i]-1]
      di+=1
  return(wtp, zw, np.argmax(zd,1), totz)


# In[ ]:

def predictDoc(wtp,wd,docs):

  for i in range(wtp.shape[0]):
      wtp[i,:] = (wtp[i,:]+1) / (wtp[i,:].sum()+wtp.shape[1])

  #print wtp
  #wd = rdf.word
  zd = np.zeros(len(docs))
  di = 0
  prob = np.zeros((len(docs),wtp.shape[0]))
  for d in docs:
    #prob = np.zeros(wtp.shape[0])
    tmp1 = wtp[:,wd[d[0]:d[1]+1]-1]
    #print d, tmp1.shape
    if d[1]-d[0]>0:
      prob[di,:] += np.sum(np.log(tmp1),1)
    else:
      prob[di,:] += np.log(tmp1)
               
    zd[di]=np.argmax(prob[di,:])
    di+=1
  return (zd, prob)


def visualTopic(T, fpath, dataset, wtp, activities=[]):
  
  p = [0,0.01,0.1,0.3,0.5,1.0]
  color = ['w','g','c','m','r']
  lines = []
  path = './data/'+dataset+'/'
  
  for j in range(2,len(p)-1):
    lines.append(mlines.Line2D([], [], color=color[j], marker='o',\
                               markersize=3,mew=0, label = str(p[j])+'< p <='+str(p[j+1])))
  for k in range(T):
      floorplan = plt.imread(path+dataset+'.jpg')
      plt.figure(frameon=False)
      plt.imshow(floorplan)
      sensor_coor = pd.read_csv(path+'sensor_coor.csv',header=0)
      plt.scatter(sensor_coor.X,sensor_coor.Y,c='w')
      V1 = wtp.shape[1]/2
      wtp_k = wtp[k,:]/float(np.max(wtp[k,:]))
      topic_sensors = pd.DataFrame(columns = ['sensor','reading','scale'])
      for j in range(2,len(p)-1):
        
          effect=np.nonzero((p[j] < wtp_k)&(wtp_k <= p[j+1]))[0]
          for i in range(len(effect)):
              sensor=effect[i]
              if sensor < V1:
                sensor_cor = sensor_coor.loc[sensor]
                reading = 'ON'
              else:
                sensor_cor = sensor_coor.loc[sensor-V1]
                reading = 'OFF'

              #print sensor_cor.sensor, reading
              topic_sensors.loc[len(topic_sensors)]= [sensor_cor.sensor, reading, wtp_k[sensor]]
              plt.plot((sensor_cor.X),(sensor_cor.Y),color[j]+'o',ms=3,mew=0)

      plt.legend(handles=lines,loc=(0,0.8),fontsize=5)
      plt.axis('off')
      if(len(activities)==0):
        plt.title(dataset+'_Topic'+str(k+1),fontsize = 8, y = 0.9)
        plt.savefig(fpath+'topic'+str(k+1)+'.png',bbox_inches='tight', pad_inches=0,dpi=300)
        topic_sensors.to_csv(fpath+'topic'+str(k+1)+'.csv',index=False,encoding='utf-8')
      else:
        plt.title(dataset+'_'+activities[k],fontsize = 8, y = 0.9)
        plt.savefig(fpath+activities[k]+'.png',bbox_inches='tight', pad_inches=0,dpi=300)
        topic_sensors.to_csv(fpath+activities[k]+'.csv',index=False,encoding='utf-8')
      plt.close()



