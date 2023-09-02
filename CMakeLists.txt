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
