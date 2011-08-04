# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


MGPU_REGISTER_BENCHMARK(micro.devicecopy devicecopy.cu)
MGPU_REGISTER_BENCHMARK(micro.invoke invoke.cpp)
MGPU_REGISTER_BENCHMARK(micro.meminfo meminfo.cu)
MGPU_REGISTER_BENCHMARK(micro.reduce reduce.cu)
