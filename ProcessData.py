#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:37:50 2017

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""

#import pandas as pd
#
#
#Types = pd.read_json(path_or_buf='types.json',orient='split') 

import json
import pandas as pd
from collections import defaultdict
import cPickle as pkl

types_wiki = json.load(open('types.json'))#type information loaded from wikidata


#with open('relation2id.txt') as fp:
#    lines = [line.strip('\n') for line in fp.readlines()]
#    r2id = dict(zip(lines[0],lines[1]))

_r2id = pd.read_table('relation2id.txt',header=None,index_col=0)
r2id = _r2id.T.to_dict('index')[1]

_e2id = pd.read_table('entity2id.txt',header=None,index_col=0)
e2id = _e2id.T.to_dict('index')[1]

__relations = pd.read_table('train.txt',header=None)
_head = [e2id[ent] for ent in __relations[0].tolist()]
_tail = [e2id[ent] for ent in __relations[1].tolist()]
_relations = [r2id[rel] for rel in __relations[2].tolist()]

relations = zip(_head,_relations,_tail) #list of tuples of form (h,r,t)

#using 15k version
fb2wiki = json.load(open('freebase_wiki_map.json'))
wiki2fb = dict([[v,k] for k,v in fb2wiki.items()])

##using 1M version 
#_fb2wiki = pd.read_table('fb_wiki_entityname.txt',header=None)[[0,1]]
#_wiki = _fb2wiki[1].tolist()
#_wiki = [unicode(ent.strip('> .').split('/')[-1],'utf-8') for ent in _wiki]
#_fb = _fb2wiki[0].tolist()
#_fb = [unicode(ent.strip('>').split('/')[-1].split('.')[0]+'/'+ent.strip('>').split('/')[-1]/split(),'utf-8') for ent in _fb]

types_fb = []

for key,value_list in types_wiki.items():
    valid_ents = []
    for ent in value_list:
        if ent in wiki2fb.keys():
            valid_ents.append(e2id[wiki2fb[ent]])
    types_fb.append(valid_ents)
        

#save the final types list. each element of the list represents 
#a type which contains all the entities belonging to that type
with open('types_fb.json','w') as fp:
    fp.write(json.dumps(types_fb,fp))


#save the set of relations in the form (h,r,t) tuples    
with open('relations_hrt.json','w') as fp:
    fp.write(json.dumps(relations,fp))
    
#make a dictionary of relations
relations_dic_h = defaultdict(list)
relations_dic_t = defaultdict(list)
for h,r,t in relations:    
        relations_dic_h[h].append((r,t))
        relations_dic_t[t].append((h,r))        
    
#save the set of relations in the form of dictionaries
with open('relations_dic_h.pkl','w') as fp:
    pkl.dump(relations_dic_h,fp)
with open('relations_dic_t.pkl','w') as fp:
    pkl.dump(relations_dic_t,fp)