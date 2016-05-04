
# coding: utf-8

# In[1]:

import pandas as pd
import scipy as sp
import numpy as np
import csv
import sys
import os
import math
import random as rd
import matplotlib.pyplot as plt


# In[3]:

import TPSN_util as util

# In[5]:

MODELS = {}
MODELS['LDA']=False
MODELS['BTM']=False
MODELS['ADL_TM']=True

#dataset='hh122'

min_durantion = 30


T = 12
ITER = 50

args = sys.argv
if len(args) < 6:
  print 'usage: test_model.py [model name] [number of Topics] [ITER] [t_th] [dataset]'
  sys.exit()
model_name = args[1]
T = int(args[2])
ITER = int(args[3])
min_durantion = int(args[4])
dataset = args[5]

print "time threshold is "+str(min_durantion)


print model_name,T,ITER,dataset

LW = MODELS[model_name]
if model_name == 'LDA':
  import LDA as model
elif model_name == 'BTM':
  import BTM as model
elif model_name == 'ADL_TM':
  import ADL_TM as model

print model_name

# In[6]:

path = './data/'+dataset+'/'


# In[7]:

rdf = pd.read_csv(path+'raw_filtered.csv',header=0)
group = pd.read_csv(path+'sensor_group.csv',header=0)


# In[8]:

activities = pd.read_csv(path+'acts.csv',header=0)
activities = activities.columns
activities

na = 0
for i in range(len(activities)):
  if activities[i] == 'No_Annotation':
    na = i+1
    break
#print "na", na

# In[9]:

rdf['group']= np.array([0]*len(rdf))
for i in range(len(group.sensor.unique())):
    rdf.loc[rdf.sensor==i+1,'group'] = int(group.loc[i,'group'])

# In[10]:

rdf = util.transformWord(rdf)
docs = util.segDocsByGrps(rdf,min_durantion)
V = max(rdf.word)
print 'V =',V
if LW:
  lwd = rdf.word
  L = max(lwd)
  lidf = np.ones(L)
else:
  lidf = []
  lwd = []

# In[16]:


if model_name == 'LDA':
    widf = np.ones(V)
else:
    widf = np.ones((V,V))



# In[52]:

(wtp, lwtp, zd, totz) = model.GibbsSampler(widf, docs, rdf.word, T, ITER, lidf,lwd)


# In[55]:

fpath = path+model_name+'_T'+str(T)+'s'+str(min_durantion)+'/'

if not os.path.exists(fpath):
  os.makedirs(fpath)

if model_name == 'LDA':
  model.visualTopic(T, fpath, dataset, wtp)
else:
  util.visualTopic(T, fpath, dataset, wtp)

np.save(fpath+'wtp.npy',wtp)
if(len(lwtp)>0):
  np.save(fpath+'lwtp.npy',lwtp)

# In[56]:

if LW:
  (nwtp,nlwtp) = util.nomalise(wtp, lwtp)
elif model_name != 'LDA':
  (nwtp,nlwtp) = util.nomalise(wtp)
else:
  nlwtp=[]
  nwtp=wtp

wd = rdf.word

if len(nlwtp) == 0:
  (pzd,prob) = model.predictDoc(nwtp,wd,docs)
else:
  (pzd,prob) = model.predictDoc(nwtp,nlwtp,wd,lwd,docs)

# In[189]:
rdf = util.setDocID(rdf,docs)
#print len(docs),zd.shape
rdf = util.setTopicID(rdf, docs, pzd)
rdf.to_csv(fpath+'processed.csv',index=False,encoding='utf-8')

stat=util.calcTopicStat(rdf,activities)
stat.to_csv(fpath+'topic_act_stat.csv')

stat=util.calcActStat(rdf,activities)
stat.to_csv(fpath+'act_topic_stat.csv')


# In[190]:

topics = util.getSegs(np.array(rdf.topic))



# In[191]:

acts = util.getSegs(np.array(rdf.activity))

# In[192]:
wf = open(fpath+'performance.csv','w')

# In[193]:

labeled = rdf.loc[rdf.activity!=na]


# In[194]:

labeled.index = np.arange(labeled.shape[0])


# In[196]:

topics = util.getSegs(np.array(labeled.topic))


# In[198]:

acts = util.getSegs(np.array(labeled.activity))


# In[199]:
wf.write('labeled data'+','+'fraction:'+str(len(topics)/float(len(acts)))+','+'seg err:'+str(util.segError(topics,labeled))+'\n')
wf.write('labeled data FMIndex: '+str(util.calFMIndex(labeled))+'\n')
wf.write('labeled data Precision, Recall: '+str(util.calPR(labeled)))
wf.close()

# In[125]:




# In[ ]:



