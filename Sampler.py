#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:10:55 2017

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""

import random
# =============================================================================
#  Sampler Function to generate samples for TransE Minibatch
# =============================================================================
def SampleTransEData(hDic,tDic,batch,VOCABULARY_SIZE = 14951):

    pos_h = []
    pos_r = []
    pos_t = []
    neg_h = []
    neg_r = []
    neg_t = []
    
    for e in batch:
        
        if e in hDic or e in tDic:
            if random.random()>0.5 and e in hDic:
                h = e
                r,t = random.sample(hDic[h])
                
                h_ = h
                r_ = r
                while True:
                    t_ = random.sample(range(VOCABULARY_SIZE))
                    if (r_,t_) not in hDic[h_].values():
                        break  
                    
            elif random.random()<=0.5 and e in tDic:
                t = e
                h,r = random.sample(tDic[t])
                t_ = t
                r_ = r
                while True:
                    h_ = random.sample(range(VOCABULARY_SIZE))
                    if (h_,r_) not in tDic[t_].values():
                        break                  
                
            pos_h.append(h)
            pos_r.append(r)
            pos_t.append(t)
            neg_h.append(h)
            neg_r.append(r)
            neg_t.append(t)            

    return pos_h,pos_r,pos_t,neg_h,neg_r,neg_t
            
    

