Tutorial
========

.. contents:: 
  :depth: 1
  :local:



A MGPU program deals with multiple GPUs in one system. To manage and assign work 
to each GPU, MGPU creates one thread for each device. Thus to get started, a
runtime environment ``mgpu::environment`` should be initialized first. This is
not mandatory: some of the functionality of the MGPU library can be used from a
single thread. For many of the advanced features of the library it is a
requirement to initialize an environment.

By default the environment utilizes all available devices. However, you might
only want to use a subset of all available GPUs in the system. The
environment constructor allows you to do just that, as it accepts a 
``mgpu::dev_group`` object that specifies which devices should be used.


.. literalinclude:: ../example/environment.cpp
  :language: c++
  :lines: 8-


Since it is imperative that the ``mgpu::environment`` instance outlives any
other object that relies on the environment, it is sensible to put all 
subsequent code in a separate scope. 

Wrappers around vendor API functions a located in the ``mgpu::backend`` 
namespace. While still incomplete, these wrappers aim to present a unified API
to different vendor interfaces such as CUDA and OpenCL.


Containers and Data Transfer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A major task when dealing with GPU systems and especially multi-GPU systems is
communication: moving data between host and the different device memories can
quickly become confusing. To simplify this task the MGPU library provides a set 
of containers and compatible transfer functions.


Simple Device Vector
--------------------

The following example shows a simple use of MGPU containers from a single
thread. Two vectors are allocated on different devices (if possible). The device
is switched temporarily for a scope using an instance of the 
``mgpu::dev_set_scoped`` class.

.. literalinclude:: ../example/simple_transfer.cpp
  :language: c++
  :lines: 8-
  
Data is copied from host memory to device memory, from there to another device
and then back to host memory. Notice that the interface of the ``mgpu::copy``
function is always the same, regardless of the arguments that are passed. It 
expects an input range and an output iterator. Other types can be adapted to be
compatible with the interface through type traits. The library currently ships
with traits for ``std::vector`` as well as ``boost::numeric::ublas::vector``.


Segmented Device Vector
-----------------------

On to a more involved example. The MGPU library draws many ideas from the
`Segmented Iterator concept 
<http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.101.1482>`_. 
It fits multi-GPU architectures well since such
systems are composed of segments of discontinuous memory. The most important
class in this regard is the segmented device vector or ``mgpu::seg_dev_vector``.
It enables you to allocate a large vector, cut the vector in slices and
distribute those slices across the devices of your choosing. The following 
example illustrates this feature:

.. literalinclude:: ../example/simple_seg_dev_vector.cpp
  :language: c++
  :lines: 8-
  
A segmented vector ``dev`` is allocated - it's size is the size of the
environment which is the number of devices in the system. The ``dev`` object 
is composed of multiple vectors, one vector on each device. The size of each 
vector is 1. The ``mgpu::copy`` function the fact that a segmented range is
passed and scatters the host vector across all devices. When copying
the data back, the segments are gathered back into one vector on the host. 

Since all operations that are executed in the context of an environment are 
asynchronous, it is necessary to wait for completion of the copies before the
memory can be accessed. ``mgpu::synchronize_barrier()`` helps in this case: 

* it synchronizes all devices; it blocks until all operations that were
  scheduled for a device are finished
* it inserts a barrier; no device can continue executing until all other devices
  have reaches this barrier
* it blocks the calling thread until the previous two conditions are met

In this example the vector is distributed equally across all devices. The way
the vector is split can be controlled using a different segmented vector 
constructor. A ``blocksize`` can be passed as an additional parameter. It 
specifies the minimum block size that the splitting algorithm is allowed to cut 
the vector in. Local vector size on each device is always be a multiple of the 
blocksize.

.. code-block:: cpp

    seg_dev_vector<float> dev(128, 8);

In this example the overall vector is of size 128 and it is split in chunks of 
8.


Invoking Kernels
^^^^^^^^^^^^^^^^

Copying data back and forth between memories certainly is not all you will want 
to do. We hope to simplify this so that you can focus on what matters most: 
implementing the algorithms that process the data once they are on the device.
Launching and managing kernels on a multi-GPU system can be tedious. We thus 
provide helper functions to simplify this. 

The functions ``mgpu::invoke_kernel()`` and ``mgpu::invoke_kernel_all()`` invoke 
user specified functions in the desired device thread and context. The former 
invokes a function in one device context (which device rank must be specified as 
the last parameter) and the latter invokes a function in all device contexts. 
These functions have a couple of features that come in handy when dealing with 
multiple GPUs:

* if a ``seg_dev_vector`` is passed to on of the invoke functions, the function
  called through invoke is not passed the entire segmented vector but a 
  reference to the local vector relevant to the current device; this can be 
  disabled for situations where a kernel needs to see the entire segmented 
  vector, for example for peer-to-peer memory access (this is also true for 
  segmented streams ``mgpu::seg_dev_stream`` and segmented iterators 
  ``mgpu::seg_dev_iterator``)
* the device id (hardware) or the device rank (device enumeration in the 
  environment) can be passed to the function called by invoke
* if a function is invoked for multiple devices (``mgpu::invoke_kernel_all()``)
  one of the devices can be selected to carry out a special task

The following example shows how a kernel could be called using the invoke
facilities of the MGPU library.

.. literalinclude:: ../example/invoke_kernel.cpp
  :language: c++
  :lines: 8-
  
The ``kernel_caller`` function is a stub that is invoked in the device thread
and context. It can launch the desired kernel itself on the device. Passing
the ``seg_dev_vector`` translates to a dev_range that corresponds to the local 
section of the segmented vector on the current device. The ``pass_dev_rank``
object is an indicator to insert each devices rank as an argument. To clarify: 
the call to ``invoke_kernel_all`` enqueues the ``kernel_caller`` function for 
each device with device-specific parameters.

Synchronization
^^^^^^^^^^^^^^^

The MGPU library is in two ways asynchronous. The first asynchronous layer is
the backend. Most backend-functions are asynchronous and can be synchronized
using ``synchronize*`` functions. The second layer is part of the MGPU library. 
For each device that is used, the library allocates a thread. Each thread
receives jobs through a separate work-queue that is populated by the main thread
(using copy, invoke etc. functions). ``barrier*`` functions can be used for 
synchronization on this layer. They synchronize the device work queues and 
optionally block the calling thread. There are combinations off the two 
functions provided as well. 


Running Functions in Device Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The environment of the MGPU library can not only be used to feed and manage
devices but is also useful to implement multi-threaded CPU algorithms. The 
``invoke*`` family of functions allows filling the device work queues without
the special features that are available in the ``invoke_kernel*`` functions and
gives simplified access to the MGPU threading functionality.


Advanced Data Transfer
^^^^^^^^^^^^^^^^^^^^^^

Apart from the ``mgpu::copy`` function, we provide a broadcast function. It
might be desirable to have the exact same vector on all devices. The following
example demonstrates how the broadcast function allows you to clone a simple
vector across all devices.

.. literalinclude:: ../example/broadcast.cpp
  :language: c++
  :lines: 8-

The host vector is only of size 1. It's content is broadcasted to the segmented
device vector: each segment will contain the same data after the broadcast. The
data is then gathered to host memory to access the result.


Distributed Batched FFTs
^^^^^^^^^^^^^^^^^^^^^^^^

Multi-GPU systems are well suited to parallelize batched FFT plans. The MGPU
library provides facilities to automatically distribute FFT plans across all
available devices. Single FFTs can currently not be split across devices but it
works well with batches of FFTs. The following example illustrates how 2D FFTs 
are distributed across devices.

.. literalinclude:: ../example/fft.cpp
  :language: c++
  :lines: 8-
  
15 2D FFTs each of size 128*128 are calculated in parallel on as many devices
as are available in the system. Input and output buffers are first allocated
on the host. Vectors of the same size but segmented and with the proper 
blocksize (the size of one 2D FFT) are then allocated on the devices. A FFT plan
is created - it is implicitly segmented. The forward and inverse functions can
be called to calculate the actual transform.