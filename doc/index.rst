.. mgpu documentation master file, created by
   sphinx-quickstart on Thu Jul 14 15:05:29 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: licence

====
MGPU
====

**Sebastian Schaetz**

Copyright © 2011 Sebastian Schaetz, distributed under the Boost Software 
License, Version 1.0. (See accompanying file LICENSE.txt or copy at 
`http://www.boost.org/LICENSE_1_0.txt <http://www.boost.org/LICENSE_1_0.txt>`_)


.. toctree:: 
    :maxdepth: 2
   
    gettingstarted
    tutorial
    examples
    rationale
    
Introduction
============

The MGPU library strives to simplify the implementation of high performance
applications and algorithms on multi-GPU systems. Its main goal is both to 
abstract platform dependent functions and vendor specific APIs, as well as 
simplifying communication between different compute elements. The library is
currently an alpha release containing only limited yet already useful 
functionality such as

* segmented device vector (one vector spread across different GPUs)
* communication functions (simple and adaptable copy and broadcast)
* methods to dispatch jobs to GPUs
* multi-GPU management and synchronization
* basic support for FFT and BLAS functions
* vendor-specific API abstraction (currently only CUDA support)

The library is currently tested on Ubuntu 10.04.3 LTS with gcc 4.4.3 and compute 
capability 2.0 CUDA devices.


Things to expect in the near future:

* more complete documentation/reference
* more tested platforms, Windows support
* more complete FFT/BLAS support
* reduce/reduce_all communication functions


.. Indices and tables
   ==================
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

