# -----
# Copyright 2011 Sebastian Schaetz. Distributed under the Boost
# Software License, Version 1.0. (See accompanying file
# LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
# -----


MGPU_REGISTER_UNITTEST(transfer.broadcast broadcast.cpp)
MGPU_REGISTER_UNITTEST(transfer.copy copy.cpp)
MGPU_REGISTER_UNITTEST(transfer.gather gather.cpp)
MGPU_REGISTER_UNITTEST(transfer.p2p_enabler p2p_enabler.cpp)
MGPU_REGISTER_UNITTEST(transfer.scatter scatter.cpp)
