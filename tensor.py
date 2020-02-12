# -*- coding: utf-8 -*-

import numpy as np
import operators as op


def tensor(data,needgrad=False):
    if isinstance(data,_Tensor):
        return data
    if isinstance(data,list) or isinstance(data,tuple):
        return op.comb([tensor(d) for d in data],needgrad)
    return _Tensor(data,needgrad)

class _Tensor:
    def __init__(self,data,needgrad=False,front=None):
        self._data=np.array(data)*1.0
        self._needgrad=needgrad
        self._front=front
        self._inroute=self._front is not None or self._needgrad
        if self._inroute:
            self._backtemp={'ifpre':False,'waitnum':0,'tempgrad':0}
            if self._needgrad:
                self._grad=np.zeros_like(self.getdata())*0.0
    def __getitem__(self,index):
        return op.cut(self,index)
    def __eq__(self,data):
        return tensor(self.getdata()==tensor(data).getdata())
    def __ne__(self,data):
        return tensor(self.getdata()!=tensor(data).getdata())
    def __gt__(self,data):
        return tensor(self.getdata()>tensor(data).getdata())
    def __lt__(self,data):
        return tensor(self.getdata()<tensor(data).getdata())
    def __ge__(self,data):
        return tensor(self.getdata()>=tensor(data).getdata())
    def __le__(self,data):
        return tensor(self.getdata()<=tensor(data).getdata())
    def __neg__(self):
        return op.mul(self,-1)
    def __add__(self,data):
        return op.add(self,data)
    def __radd__(self,data):
        return op.add(data,self)
    def __sub__(self,data):
        return op.sub(self,data)
    def __rsub__(self,data):
        return op.sub(data,self)
    def __mul__(self,data):
        return op.mul(self,data)
    def __rmul__(self,data):
        return op.mul(data,self)
    def __div__(self,data):
        return op.div(self,data)
    def __rdiv__(self,data):
        return op.div(data,self)
    def __pow__(self,data):
        return op.pow(self,data)
    def __rpow__(self,data):
        return op.pow(data,self)
    def __repr__(self):
        r=self.getdata().__repr__()
        if len(self.size())==0:
            return r
        return 'tensor'+r[5:]
    def dot(self,data):
        return op.dot(self,data)
    def zerograd(self):
        if self._needgrad:
            self._grad=np.zeros_like(self.getdata())*0.0
    def sum(self,axis=None):
        return op.sum(self,axis)
    def mean(self,axis=None):
        return op.mean(self,axis)
    def expand(self,size):
        return op.expand(self,size)
    def transpose(self,axises):
        return op.transpose(self,axises)
    def swapaxis(self,axis1,axis2):
        return op.swapaxis(self,axis1,axis2)
    def getdata(self):
        return self.getdata()
    def size(self):
        return list(self.getdata().shape)
    def _clearback(self):
        self._backtemp={'ifpre':False,'waitnum':0,'tempgrad':0}
    def reshape(self,size):
        return op.reshape(self,size)
    def getgrad(self):
        if self._needgrad:
            return self._grad
        else:
            return np.zeros_like(self.getdata())
    def needgrad(self):
        return self._needgrad
    def inroute(self):
        return self._inroute
    def backward(self):
        if not self._inroute:
            return None
        if len(self.getdata().size)>1:
            raise ValueError('only scalar can do backward')
        if not self._backtemp['ifpre']:
            self._preback()
        self._addgrad(np.ones_like(self.getdata())*1.0)
    def _preback(self):
        if not self._inroute:
            return None
        if not self._backtemp['ifpre']:
            self._backtemp['ifpre']=True
            self._backtemp['waitnum']=0
            self._backtemp['tempgrad']=np.zeros_like(self.getdata())*0.0
            if self._front is not None:
                self._front._preback()
        self._backtemp['waitnum']+=1
    def _addgrad(self,grad):
        if not self._inroute:
            return None
        self._backtemp['waitnum']-=1
        self._backtemp['tempgrad']+=grad
        if self._backtemp['waitnum']<=0:
            self._backward()
    def _backward(self):
        if self._front is not None:
            self._front._backward(self._backtemp['tempgrad'])
        if self._needgrad:
            self._grad+=self._backtemp['tempgrad']
        self._clearback()