Examples
========

.. contents:: 
  :depth: 1
  :local:
  
axpy Function
^^^^^^^^^^^^^

This example illustrates how an axpy function could be implemented. The axpy 
function calculates   

.. math:: y = y + (a*x).


Please note
that there exists an axpy function that is based on the CUDA BLAS library. This
is an example that illustrated how all the features of the MGPU library work
together: containers, communication, kernel invocation and synchronization.

First up is the kernel and a thin kernel caller function. The kernel caller
takes as an argument two device ranges ``X`` and ``Y`` and a constant ``a``. The
kernel caller calculates the correct number of threads and blocks that are
required to calculate the result. In this example the number of threads per
block is fixed to ``256``. The number of blocks is calculated from the vector
size. The caller passes ``a``, two raw pointers and the size of the vectors
to the kernel.
  
.. literalinclude:: ../example/axpy.cu
  :language: c++
  :lines: 29-45 

The kernel itself calculates the array position ``i`` it has to work with,
checks that it is not greater than the vector size and performs the calculation.
The following section shows the main function that prepares the data and invokes
the kernel. 

.. literalinclude:: ../example/axpy.cu
  :language: c++
  :lines: 51-76 
  
The MGPU library is compatible with 
`Boost.Numeric <http://www.boost.org/doc/libs/release/libs/numeric>`_ 
containers. We show the compatibility here and also use it to calculate a gold
result on the host to verify the device code. First we create the X and Y 
vectors and a third vector to store the host result ``Y_gold``. We fill the
vectors with random numbers and transfer them to equivalent segmented device
vectors ``X_dev`` and ``Y_dev``. Note that the use of the segmented device
vector automatically distributes the vector across all devices. Next we invoke
the kernel caller. We pass the function we want to invoke and its arguments to
the ``invoke_kernel_all`` function. This will call the function for each device
in each device thread and context. The segmented device vector function 
arguments are transformed to local device ranges - each range contains the 
local vector. After the computation is finished we transfer the result back to
host memory and we call ``synchronize_barrier`` to make sure all operations
finished and the data was copied to host memory. We finally perform the same 
computation on the host and compare the results.

The full code can be found in the example directory in file ``axpy.cu``.