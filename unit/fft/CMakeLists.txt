# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


MGPU_REGISTER_UNITTEST_CUFFT(fft.backend backend.cpp)
MGPU_REGISTER_UNITTEST_CUFFT(fft.calculation calculation.cpp)
MGPU_REGISTER_UNITTEST_CUFFT(fft.fft fft.cpp)
