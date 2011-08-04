# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


MGPU_REGISTER_UNITTEST(core.dev_group dev_group.cpp)
MGPU_REGISTER_UNITTEST(core.dev_memory dev_memory.cpp)
MGPU_REGISTER_UNITTEST(core.dev_ptr_pack dev_ptr_pack.cpp)
MGPU_REGISTER_UNITTEST(core.ref ref.cpp)
MGPU_REGISTER_UNITTEST(core.invoke_kernel invoke_kernel.cpp)
MGPU_REGISTER_UNITTEST(core.invoke invoke.cpp)
MGPU_REGISTER_UNITTEST(core.environment environment.cpp)
MGPU_REGISTER_UNITTEST(core.splitter splitter.cpp)
MGPU_REGISTER_UNITTEST(core.synchronization synchronization.cpp)
