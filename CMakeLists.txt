# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)

project(osqp-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()

# Eigen is usually available as a system package, so we use find_package.
find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)

# We follow
# https://github.com/abseil/abseil-cpp/blob/master/CMake/README.md#incorporating-abseil-into-a-cmake-project
# and
# https://github.com/google/googletest/blob/master/googletest/README.md#incorporating-into-an-existing-cmake-project
# by assuming that abseil-cpp and googletest are downloaded into local
# subdirectories.
# TODO(ml): Look at the suggestion in the googletest docs to automatically
# download the source at build time.

# TODO(ml): We do the same for osqp since this was easy to set up, but
# consider using find_package instead if it's more convenient for users.

add_subdirectory(abseil-cpp)

add_subdirectory(osqp)

add_subdirectory(googletest)
include(GoogleTest)

include_directories(include .)

add_library(osqp-cpp src/osqp++.cc)

target_link_libraries(osqp-cpp absl::strings absl::status Eigen3::Eigen osqpstatic dl)

# Build and register tests.
add_executable(osqp_test test/osqp++_test.cc)
target_link_libraries(osqp_test gtest gmock gtest_main absl::status absl::span Eigen3::Eigen osqp-cpp)
gtest_discover_tests(osqp_test)
