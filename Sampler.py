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
        
        if e in hDic and e in tDic:
            if random.random()>0.5:
                h = e
                while True:
                    r,t = random.sample(hDic[h],1)[0]
                    if t in batch:
                        #need to ensure sampling entities 
                        #only from batch for better convergence                        
                        break
                
                h_ = h
                r_ = r
                while True:
                    #need to ensure sampling entities 
                    #only from batch for better convergence                                            
                    t_ = random.sample(batch,1)[0]
                    if (r_,t_) not in hDic[h_]:
                        break  
                    
            else:
                t = e
                while True:
                    h,r = random.sample(tDic[t],1)[0]
                    if h in batch:
                        break
                    
                t_ = t
                r_ = r
                while True:
                    h_ = random.sample(batch,1)[0]
                    if (h_,r_) not in tDic[t_]:
                        break    

            
        elif e in hDic and e not in tDic:
            h = e
            while True:
                r,t = random.sample(hDic[h],1)[0]
                if t in batch:
                    break
            
            h_ = h
            r_ = r
            while True:
                t_ = random.sample(batch,1)[0]
                if (r_,t_) not in hDic[h_]:
                    break
            
        elif e in tDic and e not in hDic:               
            t = e
            while True:
                h,r = random.sample(tDic[t],1)[0]
                if h in batch:
                    break
            t_ = t
            r_ = r
            while True:
                h_ = random.sample(batch,1)[0]
                if (h_,r_) not in tDic[t_]:
                    break 

        pos_h.append(h)
        pos_r.append(r)
        pos_t.append(t)
        neg_h.append(h_)
        neg_r.append(r_)
        neg_t.append(t_)                                          
    return pos_h,pos_r,pos_t,neg_h,neg_r,neg_t
            
    

