# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:24:19 2020

@author: Dominic
"""
import pickle
import warnings

class Tuning_dict(dict):
    def __init__(self,*args,**kwargs):
        super(Tuning_dict,self).__init__(*args,**kwargs)
        
    def get(self,*var):
        return tuple(self[k] for k in var)
    
    def getd(self,*var):
        return dict(tuple((k,self[k]) for k in var))
    
    def getl(self,*var):
        cand = [self[k] for k in var]
        return tuple(c[-1] if isinstance(c,list) else c for c in cand)
 
    
    def app(self,**kwargs):
        dict_return = {}
        for key,item in kwargs.items():
            try:
                self[key].append(item)
                dict_return[key] = self[key]
            except AttributeError:
                raise warnings.warn("%s is not a list")
        return dict_return
    
    def add(self,**kwargs):
        for key,item in kwargs.items():
            self[key] = item
            
    def save(self,track = None, file_pth=None):
        if file_pth is None:
            file_pth = self['save_dir']+self['file_name']
            
        save_dict = self.getd(*track) if track is not None else self

        with open(file_pth,'wb') as h:
            pickle.dump(save_dict,h)
            
        return self.getl(*track)