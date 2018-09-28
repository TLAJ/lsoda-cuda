# coding: utf-8
'''
Created on 2016/09/08

@author: Kaoru
'''
import numpy as np

#「2次元配列」を「ポインタの1次元配列」に変更する
# http://yw5aj.blogspot.jp/2015/06/pass-dynamic-2d-array-from-c-to-numpy.html
def matrix2pointer(array2d):
  return (array2d.__array_interface__['data'][0] #配列の先頭のポインタ値、それを各行のオフセットに足す
          +np.arange(array2d.shape[0])*array2d.strides[0]).astype(np.uintp)

class ODEModel(object): #objectを継承しないとsuperが機能しない
  '''
  classdocs
  '''
  name_compartment = None #クラス引数は、継承先でのオーバーライドの後、継承元のコードの結果を変える
  name_param = None
  num_flux = 0
  num_free_param = 0
  
  def __init__(self): #オブジェクト引数は、継承元の動作の後、継承先で書き換えられる
    self.param_lb = np.ones(self.num_free_param,dtype=np.float64)*1.0e-6 #デフォルト上・下限
    self.param_ub = np.ones(self.num_free_param,dtype=np.float64)*1.0e4
    self.num_compartment = len(self.name_compartment) #代入する相手にはクラス変数を使わない
    self.num_param = len(self.name_param)

