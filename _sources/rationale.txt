Rationale
=========

.. contents:: 
  :depth: 1
  :local:

Invoke Limitations
^^^^^^^^^^^^^^^^^^

The kernel calling subsystem uses the Boost.Bind library extensively. For this 
reason and due to C++ limitations, invoke functions take their arguments by 
reference and cannot, therefore, accept non-const temporaries or literal 
constants. This limitation in the current C++ standard is called the forwarding 
problem.


MGPU_MAX_DEVICES vs MGPU_NUM_DEVICES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The MGPU configuration provides two defines, MGPU_MAX_DEVICES and 
MGPU_NUM_DEVICES. The former is the number of devices the library supports at 
most. The latter is a number determined for the current platform at compile-time 
and is not accurate or reliable binaries are moved between systems with 
different device configurations.


Device Identification or Rank vs ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A device thread is bound to a physical device id (``mgpu::dev_id_t``). The
device bound to device thread 0, which must not be the device with id 0 is the
device rank 0 (``mgpu::dev_rank_t``).


Device Location Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All types that are only concerned with one device (all non-segmented types) are 
unaware of where they are located. They just exist somewhere, mainly in the
context they were created in. The only exception is the device pointer. You may
ask the device id from a device pointer (``mgpu::dev_ptr::dev_id()``). 
Segmented types always have rank/location information.