cmake_minimum_required(VERSION 3.12)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(morefit)

option(WITH_ROOT "Use the ROOT libraries" ON)
if(WITH_ROOT)
	#finding ROOT
	find_package(ROOT CONFIG REQUIRED COMPONENTS Minuit MathCore GenVector RooFit RooFitCore)
	add_compile_definitions(WITH_ROOT)
	message(STATUS "Building with ROOT libraries")
	message(STATUS "ROOT include directories: ${ROOT_INCLUDE_DIRS}")
	message(STATUS "ROOT libraries: ${ROOT_LIBRARIES}")
else()
	message(STATUS "Building without ROOT libraries")
endif()

#finding LLVM
find_package(Clang REQUIRED CONFIG HINTS "${CLANG_INSTALL_PREFIX}/lib/cmake/clang")
message(STATUS "Using LLVMConfig.cmake: ${LLVM_CONFIG}")
message(STATUS "LLVM version: ${LLVM_VERSION}")
message(STATUS "Using ClangConfig.cmake: ${Clang_CONFIG}")
message(STATUS "Clang version: ${Clang_VERSION}")
set(CLANG_LIBRARY_DIR "${CLANG_INSTALL_PREFIX}/lib")
message(STATUS "Clang include directories: ${CLANG_INCLUDE_DIRS}")
message(STATUS "Clang library directory: ${CLANG_LIBRARY_DIRS}")

#finding Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

#set OpenCL directories
set(OPENCL_INCLUDE_DIR "/usr/include" CACHE PATH "OpenCL include directory")
set(OPENCL_LIBRARY_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "OpenCL library directory")
message(STATUS "OpenCL include directory: ${OPENCL_INCLUDE_DIR}")
message(STATUS "OpenCL library directory: ${OPENCL_LIBRARY_DIR}")

set(MVEC_FILE_PATH "/usr/lib/x86_64-linux-gnu/libmvec.so" CACHE FILEPATH "libmvec location")
message(STATUS "libmvec location: ${MVEC_FILE_PATH}")
add_compile_definitions("MVEC_FILE_PATH=\"${MVEC_FILE_PATH}\"")

#set compiler flags
set(CMAKE_CXX_FLAGS "-g -Wall  -D CL_TARGET_OPENCL_VERSION=300 -O3 -std=c++20") #with debug information  -march=x86-64-v4

#add standalone minuit2
add_subdirectory(minuit2)

#project include directory
set(PROJECT_INCLUDE_DIR "./include" CACHE PATH "project include directory")

if(WITH_ROOT)
include_directories(${PROJECT_INCLUDE_DIR} ${ROOT_INCLUDE_DIRS} ${PROJECT_BINARY_DIR} ${CLANG_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})
link_directories(${ROOT_LIBRARIES} ${GSL_LIBRARY_DIR} ${CLANG_LIBRARY_DIR} ${OPENCL_LIBRARY_DIR})
else()
include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_BINARY_DIR} ${CLANG_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIR})
link_directories(${GSL_LIBRARY_DIR} ${CLANG_LIBRARY_DIR} ${OPENCL_LIBRARY_DIR})
endif()

#executables for morefit project
add_executable(morefit_massfit benchmarking/morefit_massfit.cc)
add_executable(morefit_kstarmumu benchmarking/morefit_kstarmumu.cc)

#link in these libraries
#only link root if necessary
if(WITH_ROOT)
target_link_libraries(morefit_massfit OpenCL Minuit2 ${ROOT_LIBRARIES} pthread clang-cpp LLVM-17 mvec m Eigen3::Eigen)
target_link_libraries(morefit_kstarmumu OpenCL Minuit2 ${ROOT_LIBRARIES} pthread clang-cpp LLVM-17 mvec m Eigen3::Eigen)
else()
target_link_libraries(morefit_massfit OpenCL pthread Minuit2 clang-cpp LLVM-17 mvec m Eigen3::Eigen)
target_link_libraries(morefit_kstarmumu OpenCL pthread Minuit2 clang-cpp LLVM-17 mvec m Eigen3::Eigen)
endif()

#roofit exectuables for benchmarking
#only build if root is available
if(WITH_ROOT)
add_executable(roofit_massfit benchmarking/roofit_massfit.cc)
add_executable(roofit_kstarmumu benchmarking/roofit_kstarmumu.cc)

target_link_libraries(roofit_massfit ${ROOT_LIBRARIES})
target_link_libraries(roofit_kstarmumu ${ROOT_LIBRARIES})
endif()

#executable for comparison studies
#only build if root is available
if(WITH_ROOT)
add_executable(comparison_massfit tests/comparison_massfit.cc)
target_link_libraries(comparison_massfit OpenCL pthread Minuit2 clang-cpp LLVM-17 mvec m Eigen3::Eigen ${ROOT_LIBRARIES})
add_executable(comparison_kstarmumu tests/comparison_kstarmumu.cc)
target_link_libraries(comparison_kstarmumu OpenCL pthread Minuit2 clang-cpp LLVM-17 mvec m Eigen3::Eigen ${ROOT_LIBRARIES})
endif()

#build test executables
if(WITH_ROOT)
add_executable(tests_massfit tests/tests_massfit.cc)
target_link_libraries(tests_massfit OpenCL pthread Minuit2 clang-cpp LLVM-17 mvec m Eigen3::Eigen ${ROOT_LIBRARIES})
add_executable(tests_kstarmumu tests/tests_kstarmumu.cc)
target_link_libraries(tests_kstarmumu OpenCL pthread Minuit2 clang-cpp LLVM-17 mvec m Eigen3::Eigen ${ROOT_LIBRARIES})
endif()
