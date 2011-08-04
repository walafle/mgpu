Getting Started
===============

.. contents:: 
  :depth: 1
  :local:

Getting started with MGPU requires recent versions of the following software
packages: 

* `CMake <http://www.cmake.org/cmake/resources/software.html>`_ 
  (tested with 2.8)
* `Boost <http://www.boost.org/users/download/>`_ (tested with 1.45 and 1.47)
* `CUDA <http://developer.nvidia.com/cuda-toolkit-40>`_ (tested with 4.0)

From the Boost collection, the library Boost.Thread must be built. If you would
like to run unit tests the Boost.Test library should also be built beforehand.

The library compiles on the following platforms:

* g++ (Ubuntu 4.4.3-4ubuntu5) 4.4.3, 64bit
* Visual Studio 2010, 64bit (compilation test only)

The library was tested on the following systems:

* 8x GeForce GTX 580, 64bit Ubuntu
* 1x GeForce 9300 GE, 64bit Ubuntu


Configure and Build
^^^^^^^^^^^^^^^^^^^

MGPU utilizes the CMake build system. A convenient CMake feature are 
out-of-source builds. MGPU supports this and we encourage you to make use of it.

Browse to the directory where you would like the library to be built and run
:command:`cmake`:

.. code-block:: bash

  seb@defiant:~$ mkdir mgpubuild
  seb@defiant:~$ cd mgpubuild
  seb@defiant:~/mgpubuild$ cmake ../mgpu/


CMake will try to configure the library for your system and generate files to
build the library as well as unit tests. If for some reason CMake can not locate
Boost or CUDA, you may specify the path to those files manually in the 
:file:`CMakeLists.txt` file in the MGPU root directory using the following
directives:

.. code-block:: cmake

  set(CUDA_TOOLKIT_ROOT_DIR "/path/to/cuda/")
  set(BOOST_ROOT "/path/to/boost/")

CMake configures the library for the system it is run on. It determines the
number of devices that can be used for computation and their capabilities, in
particular if peer-to-peer communicatin is possible between multiple devices. 
This information is written to the :file:`include/mgpu/config.hpp` file in the
MGPU root folder and can be edited by hand if necessary.

After CMake finished, the library can be built calling calling ``make mgpu``
(Linux platform) or ``nmake mgpu`` (Windows). CMake can also
generate Visual Studio project files if the proper generator is chosen. More 
information about CMake generators can be found 
`here <http://www.vtk.org/Wiki/CMake_Generator_Specific_Information>`_.


Installing and Using MGPU
^^^^^^^^^^^^^^^^^^^^^^^^^

There are currently no means of installing the MGPU library. To use the library
you simply have to link the :file:`libmgpu` library and add the include path 
:file:`mgpu-root/inlucde`.


Testing MGPU
^^^^^^^^^^^^

The library ships with a self contained collection of unit tests. You may run
these to verify that everything works properly on your system. To run the tests
browse to the directory where MGPU was built and run :command:`ctest`:

.. code-block:: bash

  seb@defiant:~$ cd mgpubuild
  seb@defiant:~/mgpubuild$ ctest -j4
  
The ``-j4`` argument instructs :program:`ctest` to execute 4 tests in parallel 
and can speed up the runtime of the tests.
