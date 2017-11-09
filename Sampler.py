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

def SampleTypeWise(Type2Data,Ent2Type,Nsamples,NBatchX,hDic,tDic,\
                   VOCABULARY_SIZE=14951,PostoNegratio = 3,\
                   NUM_TYPES_BATCH = 128, NUM_TYPES=686, Min_Elems = 1):
    pos_h = []
    pos_r = []
    pos_t = []
    neg_h = []
    neg_r = []
    neg_t = []
    #firts sample the types that are there in this batch
    Cur_Types = random.sample(range(NUM_TYPES),NUM_TYPES_BATCH)
    #get data corresponding to Cur_Types
    relations = set()
    [relations.update(Type2Data[_type]) for _type in Cur_Types]
    relations = list(relations)
    #now sampe reqd no. of relations from this set    
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
        #delete this tuple from Type2Data
        for _type in list(Ent2Type[h]):
            if len(Type2Data[_type]) > Min_Elems:
                Type2Data[_type].discard((h,r,t))
        for _type in list(Ent2Type[t]):
            if len(Type2Data[_type]) > Min_Elems:
                Type2Data[_type].discard((h,r,t))   
                
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