import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda import cumath
from pycuda.elementwise import ElementwiseKernel
import numpy as np

start = drv.Event()
end = drv.Event()

N = 10**6

a = 2*np.ones(N,dtype=np.float64)

start.record()
np.exp(a)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Numpy",secs)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.zeros_like(a_gpu)

kernel = ElementwiseKernel(
   "double *a,double *b",
   "b[i] = exp(a[i]);",
    "kernel")

start.record() # start timing
kernel(a_gpu,b_gpu)
end.record() # end timing
end.synchronize()
secs = start.time_till(end)*1e-3
print("Kernel",secs)

start.record()
cumath.exp(a_gpu)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3
print("Cumath", secs)
