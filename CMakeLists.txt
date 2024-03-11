cmake_minimum_required(VERSION 3.16.0)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}")

# Check if ccache is present to speed up compilation time
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# On MacOS find clang, otherwise find gcc
if(${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
  find_program(CC_PROGRAM clang REQUIRED)
  find_program(CXX_PROGRAM clang++ REQUIRED)
else()
  find_program(CC_PROGRAM gcc REQUIRED)
  find_program(CXX_PROGRAM g++ REQUIRED)
endif()

# Set C and CXX compilers
if(CC_PROGRAM AND CXX_PROGRAM)
  set(CMAKE_C_COMPILER ${CC_PROGRAM})
  set(CMAKE_CXX_COMPILER ${CXX_PROGRAM})
endif()

# Project name
project(privateDPM)

# Set cmake variables
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${OpenMP_CXX_FLAGS} -funroll-loops -ftree-vectorize -Wno-deprecated")
set(CMAKE_CXX_FLAGS_DEBUG "-Og ${OpenMP_CXX_FLAGS} -Wno-deprecated")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

# Require OpenMP
find_package(OpenMP REQUIRED)

# Include FetchContent
# include(FetchContent)

# # Set up FetchContent
# set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_LIST_DIR}/lib)
# set(FETCHCONTENT_QUIET OFF)
# set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# # Fetch bayesmix library and check if it is found (SWITCH TO USUAL REPO WHEN UPDATED)
# message(CHECK_START "Fetching bayesmix")
# set(DISABLE_TESTS ON)
# set(DISABLE_PLOTS ON)
# set(DISABLE_EXAMPLES ON)
# set(BUILD_RUN OFF)
# FetchContent_Declare(
#   bayesmix
#   GIT_REPOSITORY "https://github.com/TeoGiane/bayesmix.git"
#   GIT_TAG "protobuf_any_extension"
# )
# FetchContent_MakeAvailable(bayesmix)
# message("-- bayesmix found at: " ${bayesmix_SOURCE_DIR})
# message("")

add_subdirectory(lib/bayesmix)

# Add proper link to math/tbb library
link_directories(${BAYESMIX_TBB_ROOT})

# Set include paths
set(INCLUDE_PATHS
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CMAKE_CURRENT_BINARY_DIR}
  ${BAYESMIX_INCLUDE_PATHS}
)

# Set link libraries
set(LINK_LIBRARIES
	pthread
  OpenMP::OpenMP_CXX
  ${BAYESMIX_LINK_LIBRARIES}
  bayesmixlib
)

# Set compile options
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

# Add sources for spatialCMC library
# target_sources(spatialCMC PUBLIC ${spatialcmc_PROTO_HDRS} ${spatialcmc_PROTO_SRCS})
# add_subdirectory(src)

add_subdirectory(privacy_experiments)