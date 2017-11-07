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
def SampleTransEData(temp_hDic,temp_tDic,hDic,tDic,batch,VOCABULARY_SIZE=14951\
                     , PostoNegratio = 3):

    pos_h = []
    pos_r = []
    pos_t = []
    neg_h = []
    neg_r = []
    neg_t = []
    
    def h_sample_without_replacing(key, temp_dic, orig_dic, temp_dic_t):
        if len(temp_dic[key]):
            r,t = random.choice(temp_dic[key])
            temp_dic[key].remove((r,t))
            temp_dic_t[t].remove((key,r))
        else:
            r,t = random.choice(orig_dic[key])
            
        return r,t
    
    def t_sample_without_replacing(key, temp_dic, orig_dic, temp_dic_h):
        if len(temp_dic[key]):
            h, r = random.choice(temp_dic[key])
            temp_dic[key].remove((h,r))
            temp_dic_h[h].remove((r,key))
        else:
            h,r = random.choice(orig_dic[key])
            
        return h,r
    
    for e in batch:
        
        if e in hDic and e in tDic:
            if random.random()>0.5:
                h = e
                r,t = h_sample_without_replacing(h,temp_hDic,hDic,temp_tDic)
                h_ = h
                r_ = r
                for _ in range(PostoNegratio):
                    while True:
                        t_ = random.sample(range(VOCABULARY_SIZE),1)[0]
                        if (r_,t_) not in hDic[h_]:
                            break  
                    pos_h.append(h)
                    pos_r.append(r)
                    pos_t.append(t)
                    neg_h.append(h_)
                    neg_r.append(r_)
                    neg_t.append(t_)    
                    
            else:
                t = e
                h,r = t_sample_without_replacing(t,temp_tDic,tDic,temp_hDic)
                t_ = t
                r_ = r
                for _ in range(PostoNegratio):
                    while True:
                        h_ = random.sample(range(VOCABULARY_SIZE),1)[0]
                        if (h_,r_) not in tDic[t_]:
                            break    
                    pos_h.append(h)
                    pos_r.append(r)
                    pos_t.append(t)
                    neg_h.append(h_)
                    neg_r.append(r_)
                    neg_t.append(t_)    
            
        elif e in hDic and e not in tDic:
            h = e
            r,t = h_sample_without_replacing(h,temp_hDic,hDic,temp_tDic)
            
            h_ = h
            r_ = r
            for _ in range(PostoNegratio):
                while True:
                    t_ = random.sample(range(VOCABULARY_SIZE),1)[0]
                    if (r_,t_) not in hDic[h_]:
                        break
                pos_h.append(h)
                pos_r.append(r)
                pos_t.append(t)
                neg_h.append(h_)
                neg_r.append(r_)
                neg_t.append(t_)                        
            
        elif e in tDic and e not in hDic:               
            t = e
            h,r = t_sample_without_replacing(t,temp_tDic,tDic,temp_hDic)
            t_ = t
            r_ = r
            for _ in range(PostoNegratio):            
                while True:
                    h_ = random.sample(range(VOCABULARY_SIZE),1)[0]
                    if (h_,r_) not in tDic[t_]:
                        break           
                pos_h.append(h)
                pos_r.append(r)
                pos_t.append(t)
                neg_h.append(h_)
                neg_r.append(r_)
                neg_t.append(t_)                                 
    return pos_h,pos_r,pos_t,neg_h,neg_r,neg_t

def SampleData(relations,Nsamples,NBatchX,hDic,tDic,VOCABULARY_SIZE=14951\
                     ,PostoNegratio = 3):
    pos_h = []
    pos_r = []
    pos_t = []
    neg_h = []
    neg_r = []
    neg_t = []
    
    temp = random.sample(relations,int(Nsamples/PostoNegratio))
    ents = set()
    
    def parallel(arg):
        h = arg[0]
        r = arg[1]
        t = arg[2]        
        if random.random()>0.5:
            h_ = h
            r_ = r
            while True:
                t_ = random.choice(range(VOCABULARY_SIZE))
                if (r_,t_) not in hDic[h_]:
                    break         
        else:
            t_ = t
            r_ = r
            while True:
                h_ = random.choice(range(VOCABULARY_SIZE))
                if (h_,r_) not in tDic[t_]:
                    break
        return h,r,t,h_,r_,t_
    
    for h,r,t in temp:
        del relations[(h,r,t)]
        ents.update(set([h,t]))
        sample = map(parallel,PostoNegratio*[(h,r,t)])
        _pos_h,_pos_r,_pos_t,_neg_h,_neg_r,_neg_t = zip(*sample)    
        pos_h.extend(_pos_h)
        pos_r.extend(_pos_r)
        pos_t.extend(_pos_t)       
        neg_h.extend(_neg_h)
        neg_r.extend(_neg_r)
        neg_t.extend(_neg_t)        
        
                
    ents = list(ents)
    X = random.sample(ents,NBatchX)
    
    return pos_h,pos_r,pos_t,neg_h,neg_r,neg_t,X
    
                
                
def Sample_without_replacement(data, SampleSize, Ssampled_list, Buffer):
    Sdata = set(data)
    
    Sdata.difference_update(Ssampled_list)
    
    if len(Sdata) >= SampleSize:
        return random.sample(Sdata,SampleSize)

    else:
        remaining = SampleSize - len(Sdata)
        return random.sample(Buffer,remaining) + list(Sdata)
    
            
    

