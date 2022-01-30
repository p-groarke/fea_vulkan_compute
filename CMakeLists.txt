include(${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.global.txt)

cmake_minimum_required (VERSION 3.21)
project(fea_vkc VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GoogleTest)

# User options
option(FEA_VKC_TESTS "Build and run tests." On)
option(FEA_VKC_BENCHMARKS "Build and run bencharks, requires tests." Off)

# TODO : Error on 32bit build.

set(CMAKE_CXX_STANDARD 17)

# Output to predictable location.
set_output_dir(${CMAKE_BINARY_DIR}/bin)

# External dependencies.
pull_conan()
find_package(TBB CONFIG REQUIRED QUIET)
find_package(Vulkan CONFIG REQUIRED QUIET)
find_package(vulkan-validationlayers CONFIG REQUIRED QUIET)
find_package(spirv-cross CONFIG REQUIRED QUIET)

set(FEA_TESTS OFF CACHE INTERNAL "")  # Forces the value
set(FEA_CPP17 On CACHE INTERNAL "")  # Forces the value
FetchContent_Declare(fea_libs
	GIT_REPOSITORY https://github.com/p-groarke/fea_libs.git
	GIT_TAG 689b9457fd49b627c5f8302e66563a4a21eb5422
)
FetchContent_MakeAvailable(fea_libs)

# Main Project
set(INCLUDE_NAME vkc)
file(GLOB_RECURSE HEADER_FILES "include/${INCLUDE_NAME}/*.hpp")
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp")

add_library(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
set_compile_options(${PROJECT_NAME} PUBLIC)
target_link_libraries(${PROJECT_NAME} PUBLIC TBB::TBB)
target_link_libraries(${PROJECT_NAME} PRIVATE
	Vulkan::Vulkan
	vulkan-validationlayers::vulkan-validationlayers
	spirv-cross::spirv-cross-core
	spirv-cross::spirv-cross-cpp
	spirv-cross::spirv-cross-reflect
	fea_libs
)

# Interface
target_include_directories(${PROJECT_NAME} PUBLIC
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(${PROJECT_NAME} PRIVATE src) # For based paths.

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files" FILES ${HEADER_FILES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${SOURCE_FILES})

# Install Package Configuration
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}_targets)

install(EXPORT ${PROJECT_NAME}_targets
	NAMESPACE ${PROJECT_NAME}::
	FILE ${PROJECT_NAME}-config.cmake
	DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/${INCLUDE_NAME}" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Unfortunately, we must add or modify an environment for the validation layers.
set(VK_LAYR_PATH)
set(VK_LAYR_DIR ${CMAKE_BINARY_DIR}/bin)
cmake_path(NATIVE_PATH VK_LAYR_DIR VK_LAYR_PATH)
if (WIN32)
	execute_process(COMMAND SETX VK_LAYER_PATH "${VK_LAYR_PATH}"
		COMMAND_ECHO STDOUT
	)
endif()


# Tests
if (${FEA_VKC_TESTS})
	# enable_testing()

	# Tests external dependencies.
	find_package(GTest CONFIG REQUIRED)
	find_package(stb CONFIG REQUIRED)

	# Test Project
	set(TEST_NAME ${PROJECT_NAME}_tests)
	file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp" "tests/*.hpp")
	add_executable(${TEST_NAME} ${TEST_SOURCES} ${TEST_HEADERS})
	set_compile_options(${TEST_NAME} PRIVATE)
	target_link_libraries(${TEST_NAME} GTest::gtest stb::stb fea_vkc fea_libs)

	source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/tests PREFIX "Source Files" FILES ${TEST_SOURCES})

	if (${FEA_VKC_BENCHMARKS})
		target_compile_definitions(${TEST_NAME} PRIVATE FEA_VKC_BENCHMARKS)
	endif()

	# gtest_discover_tests(${TEST_NAME})
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TEST_NAME})

	# Copy data and compile test shaders on build.
	set(DATA_IN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
	set(DATA_OUT_DIR ${BINARY_OUT_DIR}/data)
	set(STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamps)

	file(GLOB_RECURSE DATA_FILES "${DATA_IN_DIR}/*")

	set(STAMP_FILES "")
	foreach(FILE ${DATA_FILES})
		get_filename_component(FILENAME ${FILE} NAME)
		get_filename_component(FILE_PATH ${FILE} REALPATH DIRECTORY)
		file(RELATIVE_PATH FILE_OUTPUT_RPATH ${DATA_IN_DIR} ${FILE_PATH})
		# message("path test : ${FILE_OUTPUT_RPATH}")

		set(STAMP_FILE ${STAMP_DIR}/${FILENAME}.stamp)
		add_custom_command(
			OUTPUT ${STAMP_FILE}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${STAMP_DIR}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${DATA_OUT_DIR}
			COMMAND ${CMAKE_COMMAND} -E touch ${STAMP_FILE}
			COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILE}
			${DATA_OUT_DIR}/${FILE_OUTPUT_RPATH}
			DEPENDS ${FILE}
		)

		set (F_EXT "")
		get_filename_component(F_EXT ${FILE_OUTPUT_RPATH} EXT)
		# message("extension : ${F_EXT} ${DATA_OUT_DIR}/${FILE_OUTPUT_RPATH}")

		if(F_EXT STREQUAL ".comp")
			# message("\nDOING GLSLANG\n")
			add_custom_command(
				OUTPUT ${STAMP_FILE}
				COMMAND ${CMAKE_BINARY_DIR}/tools/glslangValidator -V ${DATA_OUT_DIR}/${FILE_OUTPUT_RPATH} -o ${DATA_OUT_DIR}/${FILE_OUTPUT_RPATH}.spv
				DEPENDS ${FILE}
				APPEND
			)
		endif()

		list(APPEND STAMP_FILES ${STAMP_FILE})
	endforeach()

	# Display shaders in IDE, in main project folder.
	file(GLOB_RECURSE SHADERS "${DATA_IN_DIR}/shaders/*.glsl"
		"${DATA_IN_DIR}/shaders/*.vert"
		"${DATA_IN_DIR}/shaders/*.frag"
		"${DATA_IN_DIR}/shaders/*.geom"
		"${DATA_IN_DIR}/shaders/*.tesse"
		"${DATA_IN_DIR}/shaders/*.tessc"
		"${DATA_IN_DIR}/shaders/*.comp"
	)

	add_custom_target(test_shaders
		SOURCES ${SHADERS}
		DEPENDS ${STAMP_FILES}
	)

	# set_target_properties(Shaders PROPERTIES FOLDER ${PROJECT_NAME}) # Pretty hacky :) Doesn't work on VS
	add_dependencies(${TEST_NAME} test_shaders)
endif()



