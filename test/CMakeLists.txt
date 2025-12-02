# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Add a custom target to run lit tests
add_custom_target(
  check-mqt-core-catalyst-plugin
  COMMAND ${Python3_EXECUTABLE} -m lit ${CMAKE_CURRENT_SOURCE_DIR} -v
  DEPENDS mqt-core-catalyst-plugin
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running the mqt-core-catalyst-plugin regression tests"
)
set_target_properties(check-mqt-core-catalyst-plugin PROPERTIES FOLDER "Tests" EXCLUDE_FROM_ALL ON)
