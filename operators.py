# -*- coding: utf-8 -*-

import tensor as ts
import numpy as np

class _Ioperator:
    def __init__(self,inputs):
        self._inputs=inputs
        self._inroute=False
        for i in self._inputs:
            if i.inroute():
                self._inroute=True
                break
    def _preback(self):
        for i in self._inputs:
            i._preback()
    def count(self,needgrad=False):
        data=self._countdata()
        return ts._Tensor(data,needgrad,self if self._inroute else None)
    def _ifsamesize(self,sizes):
        if len(sizes)<=1:
            return True
        s=sizes[0]
        for s2 in sizes[1:]:
            if s!=s2:
                return False
        return True
    def _countdata(self):
        pass
    def _backward(self,grad):
        pass
    
    
def sigmoid(data,needgrad):
    data=ts.tensor(data)
    return 1/(1+exp(-data))


def tanh(data,needgrad=False):
    data=ts.tensor(data)
    return _Tanh(data).count(needgrad)

class _Tanh(_Ioperator):
    def __init__(self,tensor):
        self._tensor=tensor
        super().__init__([self._tensor])
    def _countdata(self):
        return np.tanh(self._tensor.getdata())
    def _backward(self,grad):
        self._tensor._addgrad((1-np.tanh(self._tensor.getdata())**2)*grad)
        

def exp(data,needgrad=False):
    return _Exp(data).count(needgrad)

class _Exp(_Ioperator):
    def __init__(self,tensor):
        self._tensor=tensor
        super().__init__([self._tensor])
    def _countdata(self):
        return np.exp(self._tensor.getdata())
    def _backward(self,grad):
        return np.exp(self._tensor.getdata())*grad

def dot(data1,data2,needgrad=False):
    data1=ts.tensor(data1)
    data2=ts.tensor(data2)
    return _Dot(data1,data2).count(needgrad)

class _Dot(_Ioperator):
    def __init__(self,tensor1,tensor2):
        self._tensor1=tensor1
        self._tensor2=tensor2
        if len(self._tensor2.size())>2:
            raise ValueError('sorry i do not know how to do dot operation with size %s and %s'%(self._tensor1.size(),self._tensor2.size()))
        super().__init__([self._tensor1,self._tensor2])
    def _countdata(self):
        return self._tensor1.getdata().dot(self._tensor2.getdata())
    def _expandsize1(self,size):
        size=list(size)
        if len(size)<=1:
            return [1,1,-1]
        return [-1]+size[-2:]
    def _expandsize2(self,size):
        size=list(size)
        if len(size)>=2:
            return size
        if len(size)==0:
            return [1,1]
        if len(size)==1:
            return size+[1]
    def _backward(self,grad):
        data1=self._tensor1.getdata()
        size1=self._expandsize1(data1.shape)
        data1=data1.reshape(size1)
        data2=self._tensor2.getdata()
        size2=self._expandsize2(data2.shape)
        data2=data2.reshape(size2)
        size3=size1[:-1]+size2[-1:]
        grad=grad.reshape(size3)
        if self._tensor1.needgrad():
            self._tensor1._addgrad(grad.dot(data2.T).reshape(self._tensor1.size()))
        if self._tensor2.needgrad():
            grad=np.array([d.T.dot(g) for d,g in zip(data1,grad)]).sum(axis=0)
            self._tensor2._addgrad(grad.reshape(self._tensor2.size()))
            

def mean(data,axis=None,needgrad=False):
    data=ts.tensor(data)
    return _Mean(data,axis).count(needgrad)
    
class _Mean(_Ioperator):
    def __init__(self,tensor,axis=None):
        self._tensor=tensor
        self._axis=axis
        if self._axis is not None:
            if isinstance(self._axis,int):
                self._axis=[self._axis]
            self._size1=self._tensor.size()
            self._size2=[1 if i in self._axis else self._size1[i] for i in range(len(self._size1))]
        super().__init__([self._tensor])
    def _countdata(self):
        if self._axis is None:
            return self._tensor.getdata().mean()
        data=self._tensor.getdata()
        for a in reversed(self._axis):
            data=data.mean(axis=a)
        return data
    def _backward(self,grad):
        if self._axis is None:
            grad=np.ones_like(self._tensor.getdata())*grad
            grad=grad/grad.size
            self._tensor._addgrad(grad)
        else:
            grad=grad.reshape(self._size2)
            for s1,s2,a in zip(self._size1,self._size2,range(len(self._size1))):
                if s1!=s2:
                    grad=np.concatenate([grad]*s1,axis=a)/s1
            self._tensor._addgrad(grad)

def sum(data,axis=None,needgrad=False):
    data=ts.tensor(data)
    return _Sum(data,axis).count(needgrad)
    
class _Sum(_Ioperator):
    def __init__(self,tensor,axis=None):
        self._tensor=tensor
        self._axis=axis
        if self._axis is not None:
            try:
                self._axis=list(self._axis)
            except:
                self._axis=[self._axis]
            self._size1=self._tensor.size()
            self._size2=[1 if i in self._axis else self._size1[i] for i in range(len(self._size1))]
        super().__init__([self._tensor])
    def _countdata(self):
        if self._axis is None:
            return self._tensor.getdata().sum()
        data=self._tensor.getdata()
        for a in reversed(self._axis):
            data=data.sum(axis=a)
        return data
    def _backward(self,grad):
        if self._axis is None:
            self._tensor._addgrad(np.ones_like(self._tensor.getdata())*grad)
        else:
            grad=grad.reshape(self._size2)
            for s1,s2,a in zip(self._size1,self._size2,range(len(self._size1))):
                if s1!=s2:
                    grad=np.concatenate([grad]*s1,axis=a)
            self._tensor._addgrad(grad)
    
def swapaxis(data,axis1,axis2,needgrad=False):
    data=ts.tensor(data)
    axises=list(range(len(data.size())))
    axises[axis1]=axis2
    axises[axis2]=axis1
    return transpose(data,axises,needgrad)

def transpose(data,axises,needgrad=False):
    data=ts.tensor(data)
    return _Transpose(data,axises).count(needgrad)

class _Transpose(_Ioperator):
    def __init__(self,tensor,axises):
        self._tensor=tensor
        self._axises1=list(axises)
        if len(self._tensor.size())!=len(self._axises1):
            raise ValueError('can not transpose without same axis number')
        if set(self._axises1)!=set(range(len(self._tensor.size()))):
            raise ValueError('can not transpose with axises %s'%self._axises1)
        self._axises2=self._resetaxis(self._axises1)
        super().__init__([self._tensor])
    def _countdata(self):
        return self._tensor.getdata().transpose(self._axises1)
    def _backward(self,grad):
        grad=grad.transpose(self._axises2)
        self._tensor._addgrad(grad)
    def _resetaxis(self,axises):
        result=[]
        for i in range(len(axises)):
            result.append(axises.index(i))
        return result
    
    
    

    
    
def add(data1,data2,needgrad=False):
    data1=ts.tensor(data1)
    data2=ts.tensor(data2)
    return _Add(data1,data2).count(needgrad)
    
class _Add(_Ioperator):
    def __init__(self,tensor1,tensor2):
        self._tensor1,self._tensor2=boardcast(tensor1,tensor2)
        super().__init__([self._tensor1,self._tensor2])
    def _countdata(self):
        return self._tensor1.getdata()+self._tensor2.getdata()
    def _backward(self,grad):
        self._tensor1._addgrad(grad)
        self._tensor2._addgrad(grad)
        
def sub(data1,data2,needgrad=False):
    data1=ts.tensor(data1)
    data2=ts.tensor(data2)
    return _Add(data1,data2).count(needgrad)
    
class _Sub(_Ioperator):
    def __init__(self,tensor1,tensor2):
        self._tensor1,self._tensor2=boardcast(tensor1,tensor2)
        super().__init__([self._tensor1,self._tensor2])
    def _countdata(self):
        return self._tensor1.getdata()-self._tensor2.getdata()
    def _backward(self,grad):
        self._tensor1._addgrad(grad)
        self._tensor2._addgrad(-grad)
        
        
def mul(data1,data2,needgrad=False):
    data1=ts.tensor(data1)
    data2=ts.tensor(data2)
    return _Mul(data1,data2).count(needgrad)
    
class _Mul(_Ioperator):
    def __init__(self,tensor1,tensor2):
        self._tensor1,self._tensor2=boardcast(tensor1,tensor2)
        super().__init__([self._tensor1,self._tensor2])
    def _countdata(self):
        return self._tensor1.getdata()*self._tensor2.getdata()
    def _backward(self,grad):
        self._tensor1._addgrad(grad*self._tensor2.getdata())
        self._tensor2._addgrad(grad*self._tensor1.getdata())
        

def div(data1,data2,needgrad=False):
    return mul(data1,pow(data2,-1),needgrad)
        
        
def log(data,needgrad=False):
    data=ts.tensor(data)
    return _Log(data).count(needgrad)


        
class _Log(_Ioperator):
    def __init__(self,tensor):
        self._tensor=tensor
        super().__init__([self._tensor])
    def _countdata(self):
        return np.log(self._tensor.getdata())
    def _backward(self,grad):
        return (1/self._tensor.getdata())*grad
        
def pow(data1,data2,needgrad=False):
    data1,data2=boardcast(data1,data2)
    return _Pow(data1,data2).count(needgrad)
        
class _Pow(_Ioperator):
    def __init__(self,tensor1,tensor2):
        self._tensor1=tensor1
        self._tensor2=tensor2
        super().__init__([self._tensor1,self._tensor2])
    def _countdata(self):
        return self._tensor1.getdata()**self._tensor2.getdata()
    def _backward(self,grad):
        data1=self._tensor1.getdata()
        data2=self._tensor2.getdata()
        if self._tensor1.needgrad():
            self._tensor1._addgrad(np.where(data2==0,0,(data1**(data2-1))*data2*grad))
        if self._tensor2.needgrad():
            self._tensor2._addgrad(np.log(data1)*(data1**data2)*grad)
    
    
def boardcast(data1,data2):
    data1=ts.tensor(data1)
    data2=ts.tensor(data2)
    try:
        return expand(data1,data2.size()),data2
    except:
        try:
            return data1,expand(data2,data1.size())
        except:
            raise ValueError('can not boardcast with size %s and %s'%(data1.size(),data2.size()))
    
def expand(data,size,needgrad=False):
    data=ts.tensor(data)
    size=list(size)
    if data.size()==size:
        return data
    return _Expand(data,size).count(needgrad)
    
class _Expand(_Ioperator):
    def __init__(self,tensor,size):
        self._tensor=tensor
        self._size1=self._tensor.size()
        self._size2=size
        if len(self._size1)>len(self._size2):
            raise ValueError('too big size to expand size %s to %s'%(self._size1,self._size2))
        for s1,s2 in zip(reversed(self._size1),reversed(self._size2)):
            if s1!=s2 and s1!=1:
                raise ValueError('wrong size to expand size %s to %s'%(self._size1,self._size2))
        self._size1=[1]*(len(self._size2)-len(self._size1))+self._size1
        super().__init__([self._tensor])
    def _countdata(self):
        data=self._tensor.getdata()
        data=data.reshape(self._size1)
        for s1,s2,a in zip(self._size1,self._size2,range(len(self._size1))):
            if s1!=s2:
                data=np.concatenate([data]*s2,axis=a)
        return data
    def _backward(self,grad):
        for s,a in zip(reversed(self._size1),reversed(range(len(self._size1)))):
            if s==1:
                grad=grad.sum(axis=a)
        grad=grad.reshape(self._tensor.size())
        self._tensor._addgrad(grad)
    
    
def cut(data,index,needgrad=False):
    data=ts.tensor(data)
    return _Cut(data,index).count(needgrad)
    
class _Cut(_Ioperator):
    def __init__(self,tensor,index):
        self._tensor=tensor
        self._index=index
        super().__init__([self._tensor])
    def _countdata(self):
        return self._tensor.getdata().__getitem__(self._index)
    def _backward(self,grad):
        grad2=np.zeros_like(self._tensor.getdata())
        grad2.__setitem__(self._index,grad)
        self._tensor._addgrad(grad2)


def concat(datas,axis=0,needgrad=False):
    datas=[ts.tensor(d) for d in datas]
    return _Concat(datas,axis).count(needgrad)

class _Concat(_Ioperator):
    def __init__(self,tensors,axis=0):
        self._tensors=tensors
        self._axis=axis
        super().__init__(self._tensors)
    def _countdata(self):
        return np.concatenate([t.getdata() for t in self._tensors],axis=self._axis)
    def _backward(self,grad):
        index=[slice(None) for i in range(self._axis)]
        for tensor in self._tensors:
            index2=tuple(index+[slice(0,tensor.size()[self._axis])])
            tensor._addgrad(grad.__getitem__(index2))
            index2=tuple(index+[slice(tensor.size()[self._axis],None)])
            grad=grad.__getitem__(index2)


def reshape(data,size,needgrad=False):
    data=ts.tensor(data)
    return _Reshape(data,size).count(needgrad)

class _Reshape(_Ioperator):
    def __init__(self,tensor,size):
        self._tensor=tensor
        self._size1=self._tensor.size()
        self._size2=size
        super().__init__([self._tensor])
    def _countdata(self):
        return self._tensor.getdata().reshape(self._size2)
    def _backward(self,grad):
        self._tensor._addgrad(grad.reshape(self._size1))

def comb(datas,needgrad=False):
    datas=[ts.tensor(d) for d in datas]
    return _Comb(datas).count(needgrad)
    
class _Comb(_Ioperator):
    def __init__(self,tensors):
        self._tensors=tensors
        if not self._ifsamesize([t.size() for t in self._tensors]):
            raise ValueError('tensors should have same sizes')
        super().__init__(self._tensors)
    def _countdata(self):
        return np.array([t.getdata() for t in self._tensors])
    def _backward(self,grad):
        for t,g in zip(self._tensors,grad):
            t._addgrad(g)