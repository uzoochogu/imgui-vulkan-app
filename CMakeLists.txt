cmake_minimum_required (VERSION 3.8)

project (imgui-test)

set(CMAKE_CXX_STANDARD 20)

#Vulkan project
set(ENV{VULKAN_SDK} "C:/dev/VulkanSDK")
#include_directories(${VULKAN_SDK}/include/)


#executables
add_executable (imgui-test src/imgui_test.cpp src/UseImGUI.cpp src/UseImGUI.h)
add_executable (imgui-app  src/imgui-app.cpp  src/app_design.hpp)


target_include_directories(imgui-test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(imgui-app PRIVATE ${CMAKE_SOURCE_DIR}/src)

find_package(glad CONFIG REQUIRED)
target_link_libraries(imgui-test PRIVATE glad::glad)
target_link_libraries(imgui-app PRIVATE glad::glad)

find_package(imgui CONFIG REQUIRED)
target_link_libraries(imgui-test PRIVATE imgui::imgui)
target_link_libraries(imgui-app PRIVATE imgui::imgui)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(imgui-test PRIVATE glfw)
target_link_libraries(imgui-app PRIVATE glfw)


find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)

message(STATUS "Directory: ${Vulkan_INCLUDE_DIRS}")


#Vulkan executable
add_executable (vulkan_test src/vulkan_demo.cpp)
add_executable (vulkan_app src/vulkan_app.cpp)

target_include_directories(vulkan_test PRIVATE ${Vulkan_INCLUDE_DIRS})
target_include_directories(vulkan_app PRIVATE ${Vulkan_INCLUDE_DIRS})


target_link_libraries(vulkan_test PRIVATE ${Vulkan_LIBRARIES})
target_link_libraries(vulkan_app PRIVATE ${Vulkan_LIBRARIES})

target_link_libraries(vulkan_test PRIVATE glfw)
target_link_libraries(vulkan_app PRIVATE glfw)

target_link_libraries(vulkan_test PRIVATE glm::glm)
target_link_libraries(vulkan_app PRIVATE glm::glm)

set(source "${CMAKE_SOURCE_DIR}/resources") 
set(destination "${CMAKE_CURRENT_BINARY_DIR}/resources") # or CMAKE_BINARY_DIR

if(MSVC) 
  # Needed in Visual Studio to enable file access to non-local paths
  # TargetDir is a VS macro containing path to executable directory 
  set_target_properties(vulkan_app PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$(TargetDir)" )

  # Copy compiled shaders to executable directory
  # Add post-build commands
  add_custom_command(TARGET vulkan_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/resources
        "$(TargetDir)/resources" COMMENT
        "copied resource folder from ${source} => ${destination}"
  )
elseif(UNIX)
  # For non-MSVC (Linux, macOS, etc.)
  add_custom_command(TARGET vulkan_app
  COMMAND ${CMAKE_COMMAND} -E create_symlink
  ${source} ${destination} DEPENDS ${
    destination} COMMENT
  "symbolic link resource folder from ${source} => ${destination}"
  )
else()
  # windows might not use MSVC
  add_custom_command(TARGET vulkan_app POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${source}
      ${destination} COMMENT
      "copied resource folder from ${source} => ${destination}"
  )
endif() 
