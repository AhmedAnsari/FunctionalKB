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
def SampleTransEData(hDic,tDic,batch, valid_entities, Max_Tries = 100):

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
                for try_sampling in range(Max_Tries):
                    r,t = random.sample(hDic[h],1)[0]
                    if t in valid_entities:
                        #need to ensure sampling entities 
                        #only from batch for better convergence                        
                        break
                
                h_ = h
                r_ = r
                while True:
                    #need to ensure sampling entities 
                    #only from batch for better convergence                                            
                    t_ = random.sample(valid_entities,1)[0]
                    if (r_,t_) not in hDic[h_]:
                        break  
                    
            else:
                t = e
                for try_sampling in range(Max_Tries):
                    h,r = random.sample(tDic[t],1)[0]
                    if h in valid_entities:
                        break
                    
                t_ = t
                r_ = r
                while True:
                    h_ = random.sample(valid_entities,1)[0]
                    if (h_,r_) not in tDic[t_]:
                        break    

            
        elif e in hDic and e not in tDic:
            h = e
            for try_sampling in range(Max_Tries):
                r,t = random.sample(hDic[h],1)[0]
                if t in valid_entities:
                    break
            
            h_ = h
            r_ = r
            while True:
                t_ = random.sample(valid_entities,1)[0]
                if (r_,t_) not in hDic[h_]:
                    break
            
        elif e in tDic and e not in hDic:               
            t = e
            for try_sampling in range(Max_Tries):
                h,r = random.sample(tDic[t],1)[0]
                if h in valid_entities:
                    break
            t_ = t
            r_ = r
            while True:
                h_ = random.sample(valid_entities,1)[0]
                if (h_,r_) not in tDic[t_]:
                    break 
#        print try_sampling
        pos_h.append(h)
        pos_r.append(r)
        pos_t.append(t)
        neg_h.append(h_)
        neg_r.append(r_)
        neg_t.append(t_)                                          
    return pos_h,pos_r,pos_t,neg_h,neg_r,neg_t

def Sample_without_replacement(data, SampleSize, Ssampled_list, Buffer):
    Sdata = set(data)
    
    Sdata.difference_update(Ssampled_list)
    
    if len(Sdata) >= SampleSize:
        return random.sample(Sdata,SampleSize)

    else:
        remaining = SampleSize - len(Sdata)
        return random.sample(Buffer,remaining) + list(Sdata)
    
            
    

