cmake_minimum_required (VERSION 3.8)

project (imgui-vulkan-app)

set(CMAKE_CXX_STANDARD 20)

# Vulkan project
# Set your Vulkan ENV variable.
set(ENV{VULKAN_SDK} "C:/dev/VulkanSDK")
#include_directories(${VULKAN_SDK}/include/)

# Imgui Dependencies
find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# imgui executables
add_executable (imgui-test src/imgui_test.cpp src/UseImGUI.cpp src/UseImGUI.h)
add_executable (imgui-app  src/imgui-app.cpp  src/app_design.hpp)

foreach( target IN ITEMS imgui-test imgui-app )
  message(STATUS "Configuring target: ${target}")
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(${target} PRIVATE glad::glad)
  target_link_libraries(${target} PRIVATE glfw)
  target_link_libraries(${target} PRIVATE imgui::imgui)
endforeach()

# Vulkan executables dependencies
find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)

message(STATUS "Directory: ${Vulkan_INCLUDE_DIRS}")

add_executable (vulkan_test src/vulkan_demo.cpp)
add_executable (vulkan_app src/vulkan_app.cpp)
add_executable (imgui-vulkan-app  src/imgui_vulkan_app.cpp src/imgui_impl_vulkan.cpp)

foreach( target IN ITEMS vulkan_test vulkan_app imgui-vulkan-app)
  message(STATUS "Configuring target: ${target}")
  target_include_directories(${target} PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${Vulkan_LIBRARIES})
  target_link_libraries(${target} PRIVATE glfw)
  target_link_libraries(${target} PRIVATE glm::glm)
endforeach()

target_link_libraries(imgui-vulkan-app PRIVATE imgui::imgui)

set(source "${CMAKE_SOURCE_DIR}/resources") 
set(destination "${CMAKE_CURRENT_BINARY_DIR}/resources") # or CMAKE_BINARY_DIR
set( vulkan_apps  vulkan_app imgui-vulkan-app)

if(MSVC) 
  # Needed in Visual Studio to enable file access to non-local paths
  # TargetDir is a VS macro containing path to executable directory 
  set_target_properties(${vulkan_apps} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$(TargetDir)" )

  # Copy compiled shaders to executable directory
  # Add post-build commands
  add_custom_command(TARGET ${vulkan_apps} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        "$(TargetDir)/resources" COMMENT
        "copied resource folder from ${source} => ${destination}"
  )
elseif(UNIX)
  # For non-MSVC (Linux, macOS, etc.)
  add_custom_command(TARGET ${vulkan_apps}
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${source} ${destination} DEPENDS ${
    destination} COMMENT
  "symbolic link resource folder from ${source} => ${destination}"
  )
else()
  # windows might not use MSVC
  add_custom_command(TARGET ${vulkan_apps} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${source}
      ${destination} COMMENT
      "copied resource folder from ${source} => ${destination}"
  )
endif() 
