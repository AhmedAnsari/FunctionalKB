#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:17:30 2017

@author: Ahmed Ansari
@email: ansarighulamahmed@gmail.com
"""

def Evaluate_MR(gold_list, indices, P):
    """This function evaluates the mean rank 

    Args:
        gold_t_list: list of gold entity for given h, r or r, t respectively
        indices: list of lists containing indices of ranked entities
    Returns:
        
        MR: Mean rank
            
                
    """ 
    
    MR = []
    
    def get_rank(arg):
        list2 = indices[arg[1]]
        _index = list2.index(arg[0])
        return _index+1
    
    MR.extend(P.map(get_rank, zip(gold_list, xrange(len(gold_list)))))
    return MR


        