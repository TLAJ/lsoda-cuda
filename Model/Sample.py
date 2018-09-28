# coding: utf-8
'''
Created on 2016/09/08

@author: Kaoru
'''

import numpy as np
import os
import platform
from ODEModel import ODEModel, matrix2pointer

shared_dir = os.path.dirname(__file__)
shared_object = 'sample.so'
os_platform = platform.system()

if os_platform == 'Darwin':
  shared_object = 'sample_mac.so'
elif os_platform == 'Windows':
  shared_object = 'sample_win.so'

sim_cpu = np.ctypeslib.load_library(shared_object, shared_dir+'/cpu')
sim_cpu.modelSim.argtypes = (np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ct.c_int,
                             np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ct.c_int,
                             np.ctypeslib.ndpointer(np.uintp, 1, flags='C_CONTIGUOUS'))

if os.path.isfile(shared_dir+'/gpu/'+shared_object):
  sim_gpu = np.ctypeslib.load_library(shared_object, shared_dir+'/gpu')
  sim_gpu.modelSim.argtypes = (np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ct.c_int,
                               np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ndpointer(np.float64, 1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ct.c_int,
                               np.ctypeslib.ndpointer(np.uintp, 1, flags='C_CONTIGUOUS'),
                               np.ctypeslib.ct.c_int,
                               np.ctypeslib.ct.c_int)

name_param = np.array(['k1', 'k2', 'k3'])
num_sim_param = len(name_param) #シミュレータ上の要求パラメータは9個

def getSimulation(time_1d, init_1d, param_1d, input_1d, input_time_1d, num_parallel=1,num_group=1):
  num_time = len(time_1d)
  sol_buffer = np.empty((num_time,len(init_1d)),dtype=np.float64) #初期値init_1dは並列数分が渡される
  sol_buffer.fill(np.nan) #積分できなかった時はnanにする 
  
  if (num_parallel <2):
    sim_cpu.modelSim(init_1d, param_1d, time_1d, num_time, 
                     input_1d, input_time_1d, len(input_1d), matrix2pointer(sol_buffer))
  else:
    sim_gpu.modelSim(init_1d, param_1d, time_1d, num_time, 
                     input_1d, input_time_1d, len(input_1d), matrix2pointer(sol_buffer),
                     num_parallel, num_group)
  
  return sol_buffer


class Sample(ODEModel):
  name_compartment = np.array(['A', 'B', 'C'])
  num_flux = 3
  
  def __init__(self):
    super(Sample, self).__init__()
    
    self.sim_param = np.zeros(num_sim_param, dtype=np.float64)
  
  def getTimecourse(self, time_1d, init_1d, param_1d, input_1d, input_time_1d, num_parallel=1, num_group=1):
    return getSimulation(time_1d, init_1d, self.param2simParam(param_1d,num_parallel,num_group),
                         input_1d, input_time_1d, num_parallel, num_group)
