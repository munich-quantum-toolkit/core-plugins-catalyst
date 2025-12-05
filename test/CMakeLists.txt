# Copyright (c) 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Configure lit.site.cfg.py from template
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

# Dependencies needed for lit tests
set(MQT_CATALYST_PLUGIN_TEST_DEPENDS FileCheck not mqt-core-plugins-catalyst)

# Target that just builds the dependencies (for CI build step)
add_custom_target(mqt-core-plugins-catalyst-lit-test-build-only
                  DEPENDS ${MQT_CATALYST_PLUGIN_TEST_DEPENDS})
set_target_properties(mqt-core-plugins-catalyst-lit-test-build-only PROPERTIES FOLDER "Tests")

# Target that runs the lit tests
add_lit_testsuite(
  mqt-core-plugins-catalyst-lit-test "Running the MQT core-plugins-catalyst lit tests"
  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${MQT_CATALYST_PLUGIN_TEST_DEPENDS})
set_target_properties(mqt-core-plugins-catalyst-lit-test PROPERTIES FOLDER "Tests")
