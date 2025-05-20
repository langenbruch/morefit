# MoreFit

MoreFit is a framework for unbinned maximum likelihood fits with focus on parallelism and automatic optimisation. 
MoreFit is described in [arXiv:2505.12414](https://arxiv.org/abs/2505.12414), please cite the paper if you find it useful.

## Usage
MoreFit currently is a header-only library, with the header files located in the `include` directory. 
For some basic usage examples see `morefit_massfit.cc` and `morefit_kstarmumu.cc` in the `benchmarking` directory. 

## Dependencies
MoreFit aims to be lightweight with minimal dependencies. 

Required libraries are: 
* Clang and LLVM-17
* OpenCL
* Eigen 3.3
* mvec

Optional libraries:
* ROOT

## Running on lxplus:
set gcc:

`source /cvmfs/sft.cern.ch/lcg/releases/gcc/13.1.0/x86_64-el9/setup.sh`

set ROOT if wanted:

`source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.32.00/x86_64-almalinux9.4-gcc114-opt/bin/thisroot.sh`

set Clang and OpenCL directories, as well as libmvec location:

`cmake . -DCLANG_INSTALL_PREFIX=/cvmfs/sft.cern.ch/lcg/releases/clang/17.0.1-be287/x86_64-el9 -DOPENCL_INCLUDE_DIR=/cvmfs/sft.cern.ch/lcg/contrib/cuda/12.8/x86_64-el9/include -DOPENCL_LIBRARY_DIR=/cvmfs/sft.cern.ch/lcg/contrib/cuda/12.8/x86_64-el9/lib64 -DMVEC_FILE_PATH=/usr/lib64/libmvec.so`

make:

`make -j4 VERBOSE=1`
