cmake_minimum_required(VERSION 3.9)

project(CUDATests)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH})

option(ENABLE_CPPCHECK "Enable static analysis with cppcheck" OFF)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" OFF)

if(ENABLE_CPPCHECK)
  find_program(CPPCHECK cppcheck)
  if(CPPCHECK)
    set(CMAKE_CXX_CPPCHECK
        ${CPPCHECK}
        --suppress=missingInclude
        --enable=all
        --inline-suppr
        --inconclusive
        -i
        ${CMAKE_SOURCE_DIR})
  else()
    message(SEND_ERROR "cppcheck requested but executable not found")
  endif()
endif()

if(ENABLE_CLANG_TIDY)
  find_program(CLANGTIDY clang-tidy-13)
  if(CLANGTIDY)
    set(CMAKE_C_CLANG_TIDY ${CLANGTIDY})
    set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
  else()
    message(SEND_ERROR "clang-tidy could not be located.")
  endif()
endif()

# Should be using C++17 now
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FIND_PACKAGE(CUDA REQUIRED)
IF(CUDA_FOUND)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
  LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib64)
  MESSAGE(STATUS "Found CUDA ${CUDA_VERSION_STRING}, CUDA Libraries: ${CUDA_LIBRARIES}")
  MESSAGE(STATUS "CUBLAS Libraries: ${CUDA_CUBLAS_LIBRARIES}")
ELSE (CUDA_FOUND)
  MESSAGE(FATAL_ERROR "CUDA is required for compiling this project.  Please install CUDA or re-run cmake with -i to specify the CUDA directories.")
ENDIF(CUDA_FOUND)

# --------------------------------------------------------------------
# PNG and ZLIB
#
# This macro will attempt to locate the PNG include directories, link directories and libraries. The INCLUDE_DIRECTORIES
# macro adds the resulting include directories to the search path.
# --------------------------------------------------------------------
find_package(PNG REQUIRED)
if(PNG_FOUND)
  message(STATUS "PNG Version: ${PNG_VERSION_STRING}, Includes: ${PNG_INCLUDE_DIR}, Libs: ${PNG_LIBRARIES}")
  include_directories(${PNG_INCLUDE_DIR})
else(PNG_FOUND)
  message(STATUS "Could not find the PNG Libraries!")
endif(PNG_FOUND)

# Boost
FIND_PACKAGE(Boost REQUIRED program_options)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  MESSAGE(STATUS "Found Boost Libraries in ${Boost_LIBRARY_DIR}, Version ${Boost_VERSION}")
ENDIF()


include_directories(${CMAKE_SOURCE_DIR})
CUDA_ADD_EXECUTABLE(imgTest
  ArgumentParsing.h ArgumentParsing.cpp
  imgTest/ArgsImgTest.h imgTest/ArgsImgTest.cpp
  imgTest/imgTest.cu
  )
target_link_libraries(imgTest ${Boost_LIBRARIES})
target_link_libraries(imgTest ${PNG_LIBRARY})
target_link_libraries(imgTest ${ZLIB_LIBRARY})
