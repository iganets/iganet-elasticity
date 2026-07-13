########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgANet authors
#
# This file is part of the IgANet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

cmake_minimum_required (VERSION 3.24)

include(FetchContent)

########################################################################
#
# Project: IgANet-Elasticity
#
########################################################################
project(IgANet-Elasticity)

add_library(iganet_elasticity_interface INTERFACE)
add_library(iganet::elasticity ALIAS iganet_elasticity_interface)
target_include_directories(iganet_elasticity_interface INTERFACE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_BINARY_DIR}/src
)

########################################################################
# IgANets (core)
########################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  set(IGANET_CORE_LOCAL_SOURCE_DIR
      "${PROJECT_SOURCE_DIR}/../iganet"
      CACHE PATH "Path to a local IgANet core checkout")

  if(EXISTS "${IGANET_CORE_LOCAL_SOURCE_DIR}/CMakeLists.txt")
    message(STATUS "Using local IgANet core from ${IGANET_CORE_LOCAL_SOURCE_DIR}")
    add_subdirectory("${IGANET_CORE_LOCAL_SOURCE_DIR}" iganet_core)
  else()
    set(IGANET_CORE_GIT_REPOSITORY
        "https://github.com/iganets/iganet.git"
        CACHE STRING "IgANet core git repository")
    set(IGANET_CORE_GIT_TAG
        "ober_multipatch"
        CACHE STRING "IgANet core git branch, tag, or commit")

    message(STATUS
            "Fetching IgANet core from ${IGANET_CORE_GIT_REPOSITORY} "
            "(${IGANET_CORE_GIT_TAG})")
    FetchContent_Declare(
      iganet_core
      GIT_REPOSITORY ${IGANET_CORE_GIT_REPOSITORY}
      GIT_TAG ${IGANET_CORE_GIT_TAG}
    )
    FetchContent_MakeAvailable(iganet_core)
    FetchContent_GetProperties(iganet_core)
  endif()
  
  ######################################################################
  # Set C++ standard
  ######################################################################
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

########################################################################
# Collect sources
########################################################################

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
  src/examples2D/*.cxx
  src/examples3D/*.cxx
)

foreach(source ${SOURCES})

  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PRIVATE "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")
  target_link_libraries(${name} PRIVATE iganet::core)

  target_include_directories(${name}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  
endforeach()
