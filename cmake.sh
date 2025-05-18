#!/bin/bash
#build make files
cmake .
#using 4 cores
make -j4 VERBOSE=1
