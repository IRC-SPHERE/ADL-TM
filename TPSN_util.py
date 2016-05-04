
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
import matplotlib.patches as mpatches

# function for segmentation of sensor sequence by sensor groups and
# specified minimum duration of a doc
def segDocsByGrps(rdf, min_duration):
  
  grps = np.diff(np.array(rdf.group)).nonzero()[0]

  dt = pd.Timedelta(seconds=1)

  rdf.date_time=pd.to_datetime(rdf.date_time)

  docs = []
  start = 0
  ii = 0
  while( ii < len(grps)):
      stop = grps[ii]
      if ((rdf.loc[stop,'date_time'] - rdf.loc[start,'date_time'])/dt) <= min_duration:
          #print 'less than threshold', grps[ii]
          ii+=1
          if ii == len(grps):
              #print 'touch end'
              #print docs[-1], stop, len(grps)
              docs[-1] = (docs[-1][0],stop)
              start = stop+1
      else:
          #print 'find a cut',start,stop
          docs.append((start, stop))
          ii+=1
          start = stop+1

  if(stop < len(rdf)):
      stop = len(rdf)-1
      if ((rdf.loc[stop,'date_time'] - rdf.loc[start,'date_time'])/dt) <= min_duration:
          #print 'patch end1'
          docs[-1] = (docs[-1][0],stop)
      else:
          #print 'patch end2'
          #print start, stop
          docs.append((start, stop))

  return docs

# function for visualisation of the transition matrix under each discovered topic
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
      wtp_k = wtp[k,:,:]/float(np.max(wtp[k,:,:]))
      topic_sensors = pd.DataFrame(columns = ['from','from_reading','to','to_reading','scale'])
      for j in range(2,len(p)-1):
        
          effect=np.nonzero((p[j] < wtp_k)&(wtp_k <= p[j+1]))
          for i in range(len(effect[0])):
              prev=effect[0][i]
              post = effect[1][i]
              if prev < V1:
                prev_cor = sensor_coor.loc[prev]
                prev_reading = 'ON'
              else:
                prev_cor = sensor_coor.loc[prev-V1]
                prev_reading = 'OFF'
              if post < V1:
                post_cor = sensor_coor.loc[post]
                post_reading = 'ON'
              else:
                post_cor = sensor_coor.loc[post-V1]
                post_reading = 'OFF'
              #print prev_cor.sensor, post_cor.sensor
              topic_sensors.loc[len(topic_sensors)]= [prev_cor.sensor, prev_reading, post_cor.sensor, post_reading,\
                                                  wtp_k[prev,post]]
              plt.plot((prev_cor.X,post_cor.X),(prev_cor.Y,post_cor.Y),color[j]+'o-',ms=3,mew=0)

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

# function for visualisation of the transition matrix under each discovered topic
# only in black and white
def visualTopicBW(T, fpath, dataset, wtp, activities=[]):
  
  p = [0,0.01,0.1,0.3,0.5,1.0]
  cm = plt.get_cmap('Greys',50)
  color = ['w',cm(10),cm(20),cm(30),'k']
  ls = ['None','--','--','--','-']
  lines = []
  path = './data/'+dataset+'/'
  
  for j in range(2,len(p)-1):
    lines.append(mlines.Line2D([], [], color=color[j], marker='o', linestyle=ls[j],\
                               ms=3,mew=0, label = str(p[j])+'< p <='+str(p[j+1])))
  for k in range(T):
      floorplan = plt.imread(path+dataset+'_bw.jpg')
      floorplan = 255-floorplan
      plt.figure(frameon=False)
      plt.imshow(floorplan,cmap='Greys')
      sensor_coor = pd.read_csv(path+'sensor_coor.csv',header=0)
      plt.scatter(sensor_coor.X,sensor_coor.Y,c='w')
      V1 = wtp.shape[1]/2
      wtp_k = wtp[k,:,:]/float(np.max(wtp[k,:,:]))
      topic_sensors = pd.DataFrame(columns = ['from','from_reading','to','to_reading','scale'])
      for j in range(2,len(p)-1):
        
          effect=np.nonzero((p[j] < wtp_k)&(wtp_k <= p[j+1]))
          for i in range(len(effect[0])):
              prev=effect[0][i]
              post = effect[1][i]
              if prev < V1:
                prev_cor = sensor_coor.loc[prev]
                prev_reading = 'ON'
              else:
                prev_cor = sensor_coor.loc[prev-V1]
                prev_reading = 'OFF'
              if post < V1:
                post_cor = sensor_coor.loc[post]
                post_reading = 'ON'
              else:
                post_cor = sensor_coor.loc[post-V1]
                post_reading = 'OFF'
              print prev_cor.sensor, post_cor.sensor
              topic_sensors.loc[len(topic_sensors)]= [prev_cor.sensor, prev_reading, post_cor.sensor, post_reading,\
                                                      wtp_k[prev,post]]
              plt.plot((prev_cor.X,post_cor.X),(prev_cor.Y,post_cor.Y),color=color[j], marker='o', linestyle=ls[j],ms=3,mew=0,linewidth=2)

      #plt.legend(handles=lines,loc=(-0.08,0.83),fontsize=10)
      plt.axis('off')
      if(len(activities)==0):
        #plt.title(dataset+'_Topic'+str(k+1),fontsize = 13, y = 0.9)
        plt.title('         Topic'+str(k+1),fontsize = 13, y = 0.9)
        plt.savefig(fpath+'topic'+str(k+1)+'_bw.pdf',bbox_inches='tight', pad_inches=0,dpi=300)
        #topic_sensors.to_csv(fpath+'topic'+str(k+1)+'.csv',index=False,encoding='utf-8')
      else:
        plt.title('     '+activities[k],fontsize = 13, y = 0.9)
        plt.savefig(fpath+activities[k]+'_bw.pdf',bbox_inches='tight', pad_inches=0,dpi=300)
        #topic_sensors.to_csv(fpath+activities[k]+'.csv',index=False,encoding='utf-8')
      plt.close()

# visualise Topic with hours
def visualTopicTime(T, fpath, dataset, wtp, twtp, activities=[]):
  
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
      wtp_k = wtp[k,:,:]/float(np.max(wtp[k,:,:]))
      #print float(np.sum(wtp[k,:,:]))
      twtp_k = twtp[k,:]/float(np.max(twtp[k,:]))
      topic_sensors = pd.DataFrame(columns = ['from','from_reading','to','to_reading','scale'])
      time = ''
      for j in range(2,len(p)-1):
          effect=np.nonzero((p[j] < wtp_k)&(wtp_k <= p[j+1]))
          hrs = np.nonzero((p[j] < twtp_k)&(twtp_k<= p[j+1]))[0]
          time += str(p[j])+' < P(t) <= '+str(p[j+1]) +':\n'+str(hrs)+'\n'
          #print len(effect[0])
          for i in range(len(effect[0])):
              prev=effect[0][i]
              post = effect[1][i]
              if prev < V1:
                prev_cor = sensor_coor.loc[prev]
                prev_reading = 'ON'
              else:
                prev_cor = sensor_coor.loc[prev-V1]
                prev_reading = 'OFF'
              if post < V1:
                post_cor = sensor_coor.loc[post]
                post_reading = 'ON'
              else:
                post_cor = sensor_coor.loc[post-V1]
                post_reading = 'OFF'
              #print prev_cor.sensor, post_cor.sensor
              topic_sensors.loc[len(topic_sensors)]= [prev_cor.sensor, prev_reading, post_cor.sensor, post_reading,\
                                                      wtp_k[prev,post]]
              plt.plot((prev_cor.X,post_cor.X),(prev_cor.Y,post_cor.Y),color[j]+'o-',ms=3,mew=0)
      
      x = plt.xlim()[0]
      ylim = plt.ylim()
      y = (ylim[1]-ylim[0])*0.65+ylim[0]
      #print x,y
      plt.text(x,y,time,fontsize=5)
      plt.legend(handles=lines,loc=(0,0.8),fontsize=5)
      plt.axis('off')
      if(len(activities)==0):
        plt.title(dataset+'_Topic'+str(k+1),fontsize = 8, y = 0.9)
        plt.savefig(fpath+'topic'+str(k+1)+'.svg',bbox_inches='tight', pad_inches=0,dpi=100)
        topic_sensors.to_csv(fpath+'topic'+str(k+1)+'.csv',index=False,encoding='utf-8')
      else:
        plt.title(dataset+'_'+activities[k],fontsize = 8, y = 0.9)
        plt.savefig(fpath+activities[k]+'.png',bbox_inches='tight', pad_inches=0,dpi=100)
        topic_sensors.to_csv(fpath+activities[k]+'.svg',index=False,encoding='utf-8')
      plt.close()


# calculate wtp with document frequency of labeled activities
def wtpDfForActs(rdf):
  acts = np.array(rdf.activity)
  wd = rdf.word
  V = max(wd)
  wtp_a = np.zeros((max(acts),V,V))
  dacts = np.diff(acts).nonzero()[0]
  start = 0
  for d in dacts:
    stop = d
    k = acts[start]-1
    wtp_a[k,wd[start:stop]-1,wd[start+1:stop+1]-1]+=1
    start = stop+1
  return wtp_a

# calculate wtp with term frequency of labeled activities
def wtpTfForActs(rdf):
  acts = np.array(rdf.activity)
  wd = rdf.word
  V = max(wd)
  wtp_a = np.zeros((max(acts),V,V))
  dacts = np.diff(acts).nonzero()[0]
  start = 0
  for d in dacts:
    stop = d
    k = acts[start]-1
    for i in range(start,stop):
      wtp_a[k,wd[i]-1,wd[i+1]-1]+=1
    start = stop+1
  return wtp_a

# calculate wtp with term frequency of labeled activities
def uwtpTfForActs(rdf):
  acts = np.array(rdf.activity)
  wd = rdf.word
  V = max(wd)
  wtp_a = np.zeros((max(acts),V))
  dacts = np.diff(acts).nonzero()[0]
  start = 0
  for d in dacts:
    stop = d
    k = acts[start]-1
    for i in range(start,stop):
      wtp_a[k,wd[i]-1]+=1
    start = stop+1
  return wtp_a

# calculate wtp for learnt topics, words and labeles start from 1
def calcWTP(labels, wd, MM=True, DF=False):
  V = max(wd)
  docs = getSegs(np.array(labels))
  K = max(labels)
  if MM:
    wtp = np.zeros((K,V,V))
    for d in docs:
      k = labels[d[0]]-1
      if(DF):
        wtp[k,wd[d[0]:d[1]]-1,wd[d[0]+1:d[1]+1]-1]+=1
      else:
        for i in range(d[0],d[1]):
          wtp[k,wd[i]-1, wd[i+1]-1]+=1
      
  else:
    wtp = np.zeros((K,V))
    for d in docs:
      k = labels[d[0]]-1
      if DF:
        wtp[k,d[0]:d[1]+1] += 1
      else:
        for i in range(d[0],d[1]+1):
          wtp[k,wd[i]-1]+=1
  return wtp


# transfer senosr ids and readings to sensor words
def transformWord(rdf):
  rdf['word']=np.zeros((len(rdf),1),dtype=int)
  V = max(rdf.sensor)
  for i in range(1,V+1):
    rdf.loc[(rdf.sensor==i)&(rdf.reading==1), 'word'] = i
    rdf.loc[(rdf.sensor==i)&(rdf.reading==-1), 'word'] = i+V
  return rdf


# set document id to each time point
def setDocID(rdf,docs):
  di=0
  for d in docs:
    rdf.loc[d[0]:d[1],'doc'] = di
    di+=1
  return rdf

# set topic id to each time point
def setTopicID(rdf,docs,zd):
  di=0
  for d in docs:
    rdf.loc[d[0]:d[1],'topic'] = zd[di]
    di+=1
  return rdf

# calculate IDF for tag words
def getLIDF(lwd,docs):

  N = len(docs)
  #print N
  dw = np.zeros((N,max(lwd)))
  for w in range(1,max(lwd)+1):
    di = 0
    for d in docs:
      dw[di,w-1]= sum(lwd[d[0]:d[1]] == w)
      di+=1
  idf = np.ones((max(lwd)))
  for i in range(len(idf)):
    idf[i] = math.log(float(N)/sum(dw[:,i]>0))
  return (idf, dw)

# calculate IDF for words (sensor)
def getWIDF(rdf,docs):
  
  N = len(docs)
  V = max(rdf.word)
  #print N
  dw = np.zeros((N,V,V))
  wd = rdf.word
  di=0
  for d in docs:
    for i in range(d[0],d[1]):
      dw[di,wd[i]-1,wd[i+1]-1]+=1
    di+=1
  widf = np.zeros((V,V))
  for i in range(V):
    for j in range(V):
      s = sum(dw[:,i,j]>0)
      if s>0:
        widf[i,j] = math.log(float(N)/s)
  return (widf, dw)

# calculate proportions of a activity over topics
def calcActStat(rdf, activities):
  stat = pd.DataFrame()
  if 'topic' not in rdf.columns:
    print "haven't set topic id yet"
    return
  T = int(max(rdf.topic)+1)
  comp = np.array(range(0,T),ndmin = 2)
  
  for a in rdf.activity.unique():
      tps = np.array(rdf.loc[rdf.activity == a,'topic'],ndmin=2)
      comp_m = np.repeat(comp,tps.shape[1],axis=0)
      tps_m = np.repeat(tps.transpose(),T,axis=1)
      prob = sum(comp_m==tps_m)/float(tps.shape[1])
      stat[activities[a-1]]=prob
      print activities[a-1],prob.argsort()
      print np.sort(prob)
  return stat

# calculate proportions of a topic over activities
def calcTopicStat(rdf, activities):
  stat = pd.DataFrame()
  if 'topic' not in rdf.columns:
    print "haven't set topic id yet"
    return
  T = int(max(rdf.topic)+1)
  comp = np.array(range(0,T),ndmin = 2)

  for a in rdf.activity.unique():
      tps = np.array(rdf.loc[rdf.activity == a,'topic'],ndmin=2)
      comp_m = np.repeat(comp,tps.shape[1],axis=0)
      tps_m = np.repeat(tps.transpose(),T,axis=1)
      prob = sum(comp_m==tps_m)
      stat[activities[a-1]]=prob
      
      #print activities[a-1],prob.argsort()
      #print np.sort(prob)
  for t in range(T):
      s=sum(stat.loc[t])
      if s > 0:
        stat.loc[t] /= s
  print stat
  return stat

"""
def plotActsStat(act_stat, dataset, activities):
  acts = []
  for a in activities:
    acts.append(a.replace('_',' '))
  topics = []
  for i in range(act_stat.shape[0]):
    topics.append('topic '+str(i))

  act_group = pd.read_csv('./'+dataset+'/acts_group.csv',header=0)
  agn = act_group['actName'].values
  act_stat = act_stat.loc[:, agn]
  plt.figure()
  plt.
"""


# nomalise wtp and lwtp before predict on new doc.
def nomalise(wtp,lwtp=[]):
  for i in range(wtp.shape[0]):
    if len(lwtp)>0:
      lwtp[i,:] = (lwtp[i,:]+1) / (lwtp[i,:].sum()+lwtp.shape[1])
    for j in range(wtp.shape[1]):
      wtp[i,j,:] = (wtp[i,j,:]+1) / (wtp[i,j,:].sum()+wtp.shape[2])
  return (wtp,lwtp)


# calculate segmentation errors
def segError(segs,rdf):
  num_acts = max(rdf.activity)
  comp = np.array(range(1,num_acts+1),ndmin = 2)
  err = 0
  for seg in segs:
    tmp = rdf.loc[seg[0]:seg[1]]
    acts= np.array(tmp.activity,ndmin=2)
    #print acts.shape
    if (acts.shape[1] == 0):
      continue
    comp_m = np.repeat(comp,acts.shape[1],axis=0)
    acts_m = np.repeat(acts.transpose(),num_acts,axis=1)
    #print comp_m.shape, acts_m.shape
    max_g = max(sum(comp_m==acts_m))
    err += (acts.shape[1]-max_g)
  return err/float(len(rdf))

# get segmentation error positions
def segErrorPos(segs,rdf):
  num_acts = max(rdf.activity)
  comp = np.array(range(1,num_acts+1),ndmin = 2)
  err = np.zeros(len(rdf))
  for seg in segs:
    tmp = rdf.loc[seg[0]:seg[1]]
    acts= np.array(tmp.activity,ndmin=2)
    #print acts.shape
    if (acts.shape[1] == 0):
      continue
    comp_m = np.repeat(comp,acts.shape[1],axis=0)
    acts_m = np.repeat(acts.transpose(),num_acts,axis=1)
    #print comp_m.shape, acts_m.shape
    max_g = max(sum(comp_m==acts_m))
    max_a = np.argmax(sum(comp_m==acts_m))
    #print (acts.shape[1]-max_g),max_a,len(tmp.loc[tmp.activity!=max_a+1].index)
    err[tmp.loc[tmp.activity!=max_a+1].index] = 1
    
  return err


# get segments by id sequence
# segids should be array
def getSegs(segids):
  tps = np.diff(segids)
  tps = tps.nonzero()[0]
  topics=[]
  start = 0
  for i in range(len(tps)):
    stop = tps[i]
    topics.append((start,stop))
    start = stop+1
  if start < len(segids):
    topics.append((start,len(segids)-1))
  return topics

# get middle timevalues of segs
def getSegTps(segs, rdf):
  rdf.date_time = pd.to_datetime(rdf.date_time)
  rdf['timeValue'] = rdf.date_time.apply(lambda x: x.hour + x.minute*1.0 / 60)
  tps = map(lambda d: (rdf.timeValue[d[1]]-rdf.timeValue[d[0]])/2+rdf.timeValue[d[0]], segs)
  return tps

def getSegLocs(groups, segs):
    NL = max(groups)
    dlocs=np.zeros(len(segs))
    for di in range(len(segs)):
        locs = np.repeat(np.array(groups[segs[di][0]:segs[di][1]+1],ndmin=2), NL, axis=0)
        comps = np.repeat(np.array(range(1,NL+1), ndmin=2), segs[di][1]-segs[di][0]+1, axis=0).transpose()
        dlocs[di] = np.argmax(np.sum(locs==comps,1))+1
    return dlocs

def getActLocId(a, alocs, acts, activities, rdf):
    ai = (activities==a).nonzero()[0][0]
    #acts = getSegs(rdf.activity)
    alist = np.array(map(lambda i: rdf.activity[i[0]], acts))
    comps=np.repeat(np.array(range(1,int(max(alocs))),ndmin=2).transpose(), np.sum(alist == ai+1),axis=1)
    return np.argmax(np.sum(alocs[alist == ai+1] == comps, 1))



# vectorise a categorical variable,
# w should be array and start from 1
def vectorise(w):
  V = int(max(w))
  vec = np.zeros((len(w), V))
  for i in range(V):
    vec[w==(i+1),i]=1
  return vec


# visualisation of segmentation by activities,topics or documents for one day
def visSegs(rdf, segtype,subnum, fpath):
  sub = math.floor(len(rdf)/subnum)
  start = 0
  cm = plt.get_cmap('Paired')
  for k in range(subnum):
    stop = rdf.loc[rdf.index[sub*(k+1)-1],'hour']
    tmp = rdf.loc[(rdf.hour<=stop)&(rdf.hour>=start)]
    #tsegs = np.diff(np.array(tmp.topic)).nonzero()[0]
    tsegs = np.diff(np.array(tmp[segtype])).nonzero()[0]
    asegs = np.diff(np.array(tmp.activity)).nonzero()[0]
    hrs = getSegs(np.array(tmp.hour))
    start = stop+1
    #plt.figure(figsize=(2,1))
    lines = []
    s = 0.3
    #plt.xlim(0,len(day1))
    plt.stem(asegs,np.ones(len(asegs))*s,'k-',markersize=0,markerfmt='w.')
    plt.stem(tsegs,np.ones(len(tsegs))*(-1)*s,'k-',markersize=0,markerfmt='w.')
    plt.stem(range(len(tmp)),tmp.segerr*(-1)*s,'m-',markersize=0,markerfmt='w.')
    plt.plot([0,len(tmp)],[0, 0], color=cm(20), linestyle='-', linewidth=2)
    i = 0
    for h in hrs:
        plt.plot([h[0],h[1]],[s+0.1,s+0.1],color=cm(i),linestyle='-',marker=3,markersize=12, linewidth=3)
        lines.append(mlines.Line2D([], [], color=cm(i), linestyle='-',linewidth=3,\
                                   label = 'Hour: '+str(int(tmp.loc[tmp.index[h[0]],'hour']))))
        i+=15
    plt.axis('off')
    lines.append(mlines.Line2D([], [], color='m', linestyle=':',linewidth=5,\
                               label = 'Error points'))
    lgd=plt.legend(handles=lines,loc=2,bbox_to_anchor=(1.0, 0.8),fontsize=8)
    x = plt.xlim()[0]
    ylim = plt.ylim()
    y1 = (ylim[1]-ylim[0])*0.77+ylim[0]
    y2 = ylim[0]-(ylim[1]-ylim[0])*0.06
    ttt = pd.to_datetime(rdf.date_time[rdf.index[0]])
    plt.text(x,y1,'Segmentation \n by Activities',fontsize=8)
    plt.text(plt.xlim()[1]-0.3*(plt.xlim()[1]-x),y2,'Date of data: '+str(ttt.date()),fontsize=8)
    if segtype == 'topic':
      segname = 'Topics'
    elif segtype == 'doc':
      segname = 'Documents'
    else:
      segname = 'Locations'
    txt=plt.text(x,y2,'Segmentation \n by '+ segname,fontsize=8)
    #plt.show()
    plt.savefig(fpath+'act_'+segtype+'_segs_sub'+str(k+1)+'.pdf',bbox_extra_artists=(lgd,txt,),bbox_inches='tight',transparent=True,dpi=100)
    plt.close()

#visualisation locations of a house
def visualLoc(fpath, dataset):
  circles = []
  path = './data/'+dataset+'/'
  floorplan = plt.imread(path+dataset+'.jpg')
  plt.figure(frameon=False)
  plt.imshow(floorplan)
  sensor_coor = pd.read_csv(path+'sensor_coor.csv',header=0)
  plt.scatter(sensor_coor.X,sensor_coor.Y,c='w')
  
  group = pd.read_csv(path+'sensor_group.csv',header=0)
  ng = max(group.group)
  cm = plt.get_cmap('rainbow',ng)
  locs = {}
  locats = np.array(group.locat)
  for k in range(len(group)):
    sensor_cor = sensor_coor.loc[k]
    g = int(group.loc[group.sensor == sensor_cor.sensor,'group'])
    locs[g] = locats[group.loc[group.sensor == sensor_cor.sensor].index[0]]
    plt.plot((sensor_cor.X),(sensor_cor.Y),color=cm(g-1),marker='o',ms=3,mew=0)

  for j in range(ng):
      print locs[j+1]
      c = mpatches.Circle((0, 0), 0.1, color=cm(j),label = locs[j+1])
        #circles.append(mlines.Line2D([], [], color=cm(j+51),'o',\
          #                                   markersize=3,mew=0, label = locs[j+1]))
      circles.append(c)
  lgd=plt.legend(handles=circles,loc=(-0.1,0.2),fontsize=5)
  plt.axis('off')

  plt.title(dataset+' Locations and Sensors',fontsize = 8, y = 0.9)
  plt.savefig(fpath+'groups.pdf',bbox_inches='tight',bbox_extra_artists=(lgd,),dpi=200)

  plt.close()

# transform raw data space to data space represented by activities or topics
def genEventStat(segs,type,labeled):
  col = ['day','begin','end',type,'duration']
  num_words = max(labeled.word)
  for i in range(num_words):
    col.append('word'+str(i+1))
  comp = np.array(range(1,num_words+1),ndmin = 2)
  stat = pd.DataFrame(columns=col)
  idx = 0
  for ti in segs:
    #day = labeled.loc[ti[0],'date_time'].day
    begin = labeled.loc[ti[0],'date_time']
    end = labeled.loc[ti[1],'date_time']
    
    if(end.hour >= begin.hour):
      end = end.hour * 3600 + end.minute * 60 + end.second
    else:
      end = (end.hour + 23)* 3600 + (end.minute+59) * 60 + end.second + 59
    day = begin.day
    begin = begin.hour * 3600 + begin.minute * 60 + begin.second
    
    stat.loc[idx,'day':'duration'] = [day,begin, end,labeled.loc[ti[0],type],end-begin]

    #stat.loc[idx,'day':'duration'] = [day,(labeled.loc[ti[0],'date_time']).hour, (labeled.loc[ti[1],'date_time']).hour,labeled.loc[ti[0],type],end-begin]
    
    words = np.array(labeled.loc[ti[0]:ti[1],'word'],ndmin=2)
    
    comp_m = np.repeat(comp,words.shape[1],axis=0)
    word_m = np.repeat(words.transpose(),num_words,axis=1)
    #print ti, comp_m.shape, word_m.shape,np.sum(comp_m==word_m,axis=0).shape
    stat.loc[idx,'word1':'word'+str(num_words)] = np.sum(comp_m==word_m,axis=0)/float(np.sum(comp_m==word_m))

    idx+=1
  stat.begin /= 23*3600+59*60+59
  stat.end /= 23*3600+59*60+59
  if(type == 'topic'):
    stat[type]+=1
  return stat

# visualise data space represented by activities or topics
def visEventStat(fpath,stat,type,elist,activities=[]):
  cm = plt.get_cmap('rainbow',32)
  lgd=[]
  idx = 0

  if type == 'Topic':
    cm = plt.get_cmap('gist_rainbow',30)
    for i in elist:
      estat = stat.loc[stat.topic == i]
      hdl, =plt.plot(estat.begin*24, estat.duration/3600,color=cm(i),linestyle='None', marker='o',label='Topic '+str(i))
      lgd.append(hdl)
      idx+=1
  elif type == 'Location':
      cm = plt.get_cmap('gist_rainbow',len(elist))
      for i in elist:
        estat = stat.loc[stat.group == i]
        hdl, =plt.plot(estat.begin*24, estat.duration/3600,color=cm(i),linestyle='None', marker='o',label='Location '+str(i))
        lgd.append(hdl)
        idx+=1
  elif (type == 'Activity')and(len(activities)>0) :
    for a in elist:
      aid = (activities==a).nonzero()[0][0] + 1
      print aid
      estat = stat.loc[stat.activity == aid]
      hdl, =plt.plot(estat.begin*24, estat.duration/3600,color=cm(aid),linestyle='None', marker='o',label=a)
      lgd.append(hdl)
      idx+=1
  else:
    print 'wrong input!'
    return
  
  plt.axis('tight')
  xlgd=plt.legend(handles=lgd,loc=(0.6,0.7),fontsize=15)
#plt.xticks(np.arange(0,23,2))
#plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(24))

  plt.xlabel('Start time of an occurrence (military time of a day)',fontsize=15)
  plt.ylabel('Durations (unit = 1 hour)',fontsize=15)
  plt.title(type+' Occurrences and Durations',fontsize = 15, y = 0.9)
  plt.savefig(fpath+type+'_time_stat.pdf',bbox_inches='tight',bbox_extra_artists=(xlgd,),dpi=200)
              
  plt.close()


# visualise data space represented by activities or topics
# Only in black and white
def visEventStatBW(fpath,stat,type,elist,activities=[]):
  cm = plt.get_cmap('Greys',len(elist)+50)
  lgd=[]
  idx = 0
  
  if type == 'Topic':
    cm = plt.get_cmap('Greys',len(elist)+50)
    for i in elist:
      estat = stat.loc[stat.topic == i]
      hdl, =plt.plot(estat.begin*24, estat.duration/3600,color=cm(i+10),linestyle='None', marker='o',label='Topic '+str(i))
      lgd.append(hdl)
      idx+=1
  elif type == 'Location':
      cm = plt.get_cmap('gist_rainbow',len(elist))
      for i in elist:
        estat = stat.loc[stat.group == i]
        hdl, =plt.plot(estat.begin*24, estat.duration/3600,color=cm(i),linestyle='None', marker='o',label='Location '+str(i))
        lgd.append(hdl)
        idx+=1
  elif (type == 'Activity')and(len(activities)>0) :
    for a in elist:
      aid = (activities==a).nonzero()[0][0] + 1
      print aid
      estat = stat.loc[stat.activity == aid]
      hdl, =plt.plot(estat.begin*24, estat.duration/3600,color=cm(idx+10),linestyle='None', marker='o',label=a)
      lgd.append(hdl)
      idx+=1
  else:
    print 'wrong input!'
    return

  plt.axis('tight')
  xlgd=plt.legend(handles=lgd,loc=(0.4,0.8),fontsize=15)
  plt.xticks(np.arange(0,23,2))
  plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(24))
  
  plt.xlabel('Start time of an occurrence (military time of a day)',fontsize=15)
  plt.ylabel('Durations (unit = 1 hour)',fontsize=15)
  plt.title(type+' Occurrences and Durations',fontsize = 15, y = 0.9)
  plt.savefig(fpath+type+'_time_stat_bw.pdf',bbox_inches='tight',bbox_extra_artists=(xlgd,),dpi=200)
  
  plt.close()


# visualisation of segmentation by activities,topics or documents for one day
# Only in black and white
def visSegsBW(rdf, segtype,subnum, fpath):
  sub = math.floor(len(rdf)/subnum)
  start = 0
  cm = plt.get_cmap('Greys')
  for k in range(subnum):
    stop = rdf.loc[rdf.index[sub*(k+1)-1],'hour']
    tmp = rdf.loc[(rdf.hour<=stop)&(rdf.hour>=start)]
    #tsegs = np.diff(np.array(tmp.topic)).nonzero()[0]
    tsegs = np.diff(np.array(tmp[segtype])).nonzero()[0]
    asegs = np.diff(np.array(tmp.activity)).nonzero()[0]
    hrs = getSegs(np.array(tmp.hour))
    start = stop+1
    #plt.figure(figsize=(2,1))
    lines = []
    s = 0.3
    #plt.xlim(0,len(day1))
    plt.stem(asegs,np.ones(len(asegs))*s,'k-',markersize=0,markerfmt='w.')
    plt.stem(tsegs,np.ones(len(tsegs))*(-1)*s,'k-',markersize=0,markerfmt='w.')
    #plt.stem(range(len(tmp)),tmp.segerr*(-1)*s,'m-',markersize=0,markerfmt='w.')
    plt.plot([0,len(tmp)],[0, 0], color=cm(200), linestyle='-', linewidth=2)
    i = 0
    for h in hrs:
      plt.plot([h[0],h[1]],[s+0.1,s+0.1],color=cm(i),linestyle='-',marker='.',markersize=1, linewidth=1)
      lines.append(mlines.Line2D([], [], color=cm(i), linestyle='-',linewidth=1,\
                                   label = 'Hour: '+str(int(tmp.loc[tmp.index[h[0]],'hour']))))
      i+=15
    plt.axis('off')
      #lines.append(mlines.Line2D([], [], color='m', linestyle=':',linewidth=5,\
      #                         label = 'Error points'))
    lgd=plt.legend(handles=lines,loc=2,bbox_to_anchor=(1.0, 0.8),fontsize=8)
    x = plt.xlim()[0]
    ylim = plt.ylim()
    y1 = (ylim[1]-ylim[0])*0.77+ylim[0]
    y2 = ylim[0]-(ylim[1]-ylim[0])*0.09
    ttt = pd.to_datetime(rdf.date_time[rdf.index[0]])
    plt.text(x,y1,'Segmentation \n by Activities',fontsize=15)
    plt.text(plt.xlim()[1]-0.5*(plt.xlim()[1]-x),y2,'Date of data: '+str(ttt.date()),fontsize=15)
    if segtype == 'topic':
       segname = 'Topics'
    elif segtype == 'doc':
       segname = 'Documents'
    else:
       segname = 'Locations'
    txt=plt.text(x,y2,'Segmentation \n by '+ segname,fontsize=15)
    #plt.show()
    plt.savefig(fpath+'act_'+segtype+'_segs_sub'+str(k+1)+'_bw.pdf',bbox_extra_artists=(lgd,txt,),bbox_inches='tight',transparent=True,dpi=100)
    plt.close()


#calc probability from a beta distribution
def pBeta(x, a, b):
  return math.pow(x,a-1) * math.pow(1-x, b-1) / sp.beta(a,b)



# Calculate Rand Index
def calRandIndex(labeled):
  TP = 0
  TN = 0
  FN = 0
  FP = 0
  for t in labeled.topic.unique():
    for a in labeled.activity.unique():
      tmp = labeled.loc[(labeled.topic==t)&(labeled.activity==a)].shape[0]
      tmp1 = labeled.loc[(labeled.topic!=t)&(labeled.activity!=a)].shape[0]
      tmp2 = labeled.loc[(labeled.topic!=t)&(labeled.activity==a)].shape[0]
      tmp3 = labeled.loc[(labeled.topic==t)&(labeled.activity!=a)].shape[0]
      TP += tmp * (tmp-1)
      TN += tmp * tmp1
      FN += tmp * tmp2
      FP += tmp * tmp3

  return (TP+TN)*1.0/(TP+TN+FP+FN)

def calFMIndex(labeled):
  TP = 0
  TN = 0
  FN = 0
  FP = 0
  for t in labeled.topic.unique():
    for a in labeled.activity.unique():
      tmp = labeled.loc[(labeled.topic==t)&(labeled.activity==a)].shape[0]
      tmp1 = labeled.loc[(labeled.topic!=t)&(labeled.activity!=a)].shape[0]
      tmp2 = labeled.loc[(labeled.topic!=t)&(labeled.activity==a)].shape[0]
      tmp3 = labeled.loc[(labeled.topic==t)&(labeled.activity!=a)].shape[0]
      TP += tmp * (tmp-1)
      TN += tmp * tmp1
      FN += tmp * tmp2
      FP += tmp * tmp3

  return (TP)*1.0/math.sqrt((TP+FP)*(TP+FN))

def calFMeaure(labeled):
  TP = 0
  TN = 0
  FN = 0
  FP = 0
  for t in labeled.topic.unique():
    for a in labeled.activity.unique():
      tmp = labeled.loc[(labeled.topic==t)&(labeled.activity==a)].shape[0]
      tmp1 = labeled.loc[(labeled.topic!=t)&(labeled.activity!=a)].shape[0]
      tmp2 = labeled.loc[(labeled.topic!=t)&(labeled.activity==a)].shape[0]
      tmp3 = labeled.loc[(labeled.topic==t)&(labeled.activity!=a)].shape[0]
      TP += tmp * (tmp-1)
      TN += tmp * tmp1
      FN += tmp * tmp2
      FP += tmp * tmp3

  P = TP*1.0/(TP+FP)
  R = TP*1.0/(TP+FN)
  return (P*R*2.0)/(P+R)

def calPR(labeled):
  TP = 0
  TN = 0
  FN = 0
  FP = 0
  for t in labeled.topic.unique():
    for a in labeled.activity.unique():
      tmp = labeled.loc[(labeled.topic==t)&(labeled.activity==a)].shape[0]
      tmp1 = labeled.loc[(labeled.topic!=t)&(labeled.activity!=a)].shape[0]
      tmp2 = labeled.loc[(labeled.topic!=t)&(labeled.activity==a)].shape[0]
      tmp3 = labeled.loc[(labeled.topic==t)&(labeled.activity!=a)].shape[0]
      TP += tmp * (tmp-1)
      TN += tmp * tmp1
      FN += tmp * tmp2
      FP += tmp * tmp3

  P = TP*1.0/(TP+FP)
  R = TP*1.0/(TP+FN)
  return (P,R)




