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
target_include_directories(iganet_elasticity_interface INTERFACE ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

########################################################################
# IgANets (core)
########################################################################
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
    FetchContent_Declare(
      iganet_core
      GIT_REPOSITORY https://github.com/iganets/iganet.git
    )
  else()  
    FetchContent_Declare(
      iganet_core
      URL https://github.com/IgANets/iganet/archive/refs/heads/master.zip
    )
  endif()

  set(IGANET_WITH_GISMO ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(iganet_core)
  FetchContent_GetProperties(iganet_core)
  
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
file(GLOB SOURCES src/*.cxx)

foreach(source ${SOURCES})

  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})

  if(IGANET_BUILD_PCH AND NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS "3.16.0")
    target_precompile_headers(${name} REUSE_FROM iganet_pch)
  endif()

  target_compile_options(${name} PRIVATE "-DIGANET_DATA_DIR=\"${PROJECT_SOURCE_DIR}/filedata/\"")
  target_link_libraries(${name} PRIVATE iganet::core)
  
endforeach()
