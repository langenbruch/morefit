include(CMakeFindDependencyMacro)

set(minuit2_omp OFF)
set(minuit2_mpi OFF)

if(minuit2_omp)
    find_dependency(OpenMP REQUIRED)

    if(OpenMP_FOUND OR OpenMP_CXX_FOUND)
        # For CMake < 3.9, we need to make the target ourselves
        if(NOT TARGET OpenMP::OpenMP_CXX)
            add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
            set_property(TARGET OpenMP::OpenMP_CXX
                         PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
            # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
            set_property(TARGET OpenMP::OpenMP_CXX
                         PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS})

            find_dependency(Threads REQUIRED)
        endif()
    endif()
endif()

if(minuit2_mpi)
    find_dependency(MPI REQUIRED)

    # For supporting CMake < 3.9:
    if(MPI_FOUND OR MPI_CXX_FOUND)
        if(NOT TARGET MPI::MPI_CXX)
            add_library(MPI::MPI_CXX IMPORTED INTERFACE)

            set_property(TARGET MPI::MPI_CXX
                         PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
            set_property(TARGET MPI::MPI_CXX
                         PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}") 
            set_property(TARGET MPI::MPI_CXX
                         PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
        endif()
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/Minuit2Targets.cmake")

add_library(Minuit2::Math IMPORTED INTERFACE)
set_property(TARGET Minuit2::Math
             PROPERTY INTERFACE_LINK_LIBRARIES Minuit2::Minuit2Math)

add_library(Minuit2::Common IMPORTED INTERFACE)
set_property(TARGET Minuit2::Common
             PROPERTY INTERFACE_LINK_LIBRARIES Minuit2::Minuit2Common)
