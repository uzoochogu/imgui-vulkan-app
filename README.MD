# ImGui Vulkan Application
This is a personal reference for getting started with Vulkan and ImGui (using the Vulkan backend). 
The code is heavily annotated for learning purposes.

## Structure
Each chapter of the vulkan tutorial is in branches:
1. [`00-end-of-drawing`](https://github.com/uzoochogu/imgui-vulkan-app/tree/00-end-of-drawing)
2. [`01-drawing-a-triangle`](https://github.com/uzoochogu/imgui-vulkan-app/tree/01-drawing-a-triangle)
3. [`02-vertex-buffers`](https://github.com/uzoochogu/imgui-vulkan-app/tree/02-vertex-buffers)
4. [`03-uniform-buffers`](https://github.com/uzoochogu/imgui-vulkan-app/tree/03-uniform-buffers)
5. [`04-texture-mapping`](https://github.com/uzoochogu/imgui-vulkan-app/tree/04-texture-mapping)
6. [`05-depth-buffering`](https://github.com/uzoochogu/imgui-vulkan-app/tree/05-depth-buffering)
7. [`06-loading-models`](https://github.com/uzoochogu/imgui-vulkan-app/tree/06-loading-models)
8. [`07-generating-mipmaps`](https://github.com/uzoochogu/imgui-vulkan-app/tree/07-generating-mipmaps)
9. [`08-multisampling`](https://github.com/uzoochogu/imgui-vulkan-app/tree/08-multisampling)
10. [`09-compute-shaders`](https://github.com/uzoochogu/imgui-vulkan-app/tree/09-compute-shaders)
11. `main` branch contains an imgui-vulkan-app.cpp file that integrates imgui with the resulting vulkan renderer.

## Dependency managment
These are the project dependencies
1. `imgui` with glfw, opengl and vulkan bindings
2. `glfw` - windowing
3. `glm` - linear algebra, good compatibility with GLSL.
4. `stb` - loading images
5. `tinyobjloader` - loading obj models
6. `Vulkan`

You can follow this guide for setting up dependencies:

### Setting up Vulkan
You can install Vulkan from: [`LunarG`](https://vulkan.lunarg.com/)

Note the install location. Run the `vkcube` executable in the `Bin` directory to verify a successful installation.

Copy Vulkan SDK installation path to the [`CMakeLists.txt`](./CMakeLists.txt) file.
Note that it might also be automatically installed by `vcpkg`.
### vcpkg.json
`vcpkg` is used for package and dependency management. 
A `vcpkg.json` user manifest file is provided to help install dependencies or you can install them globally.
```bash
# run this in any folder that vcpkg.json is in.
# triplet is optional, but useful on windows
path_to_vcpkg install --triplet triplet_name

# e.g
C:/dev/vcpkg/vcpkg install --triplet x64-windows
```
Global install:
```bash
path_to_vcpkg install imgui[glfw-binding,opengl3-binding,vulkan-binding]:triplet_name
path_to_vcpkg install glad:triplet_name
path_to_vcpkg install glfw3:triplet_name
path_to_vcpkg install glm:triplet_name
path_to_vcpkg install tinyobjloader[double]:triplet_name
path_to_vcpkg install stb:triplet_name
path_to_vcpkg integrate install
```

## Shaders compilation
For branches that make use of the shader code, you would need to compile them before use. Scripts for most platform have been provided to help in the compilation of the shaders. 
Consult [`the shaders readme`](./resources/shaders/readme.md) for more information.
You would need to do this for both [`shaders`](./resources/shaders/) and [`compute-shaders`](./resources/compute-shaders/) directories.
```bash
# for example
cd ./resources/shaders
./compile.bat
cd ../compute-shaders
./compile.bat

# Linux
cd ./resources/shaders
./compile
cd ../compute-shaders
./compile
```

## Building
Do the following steps
1. [`Install dependencies`](#dependency-managment)
2. [`Compile shaders`](#shaders-compilation)
3. Generate and/or Building with CMake

`CMake` is used to generate project files.
If you are working with vcpkg for dependency management, you can pass the toolchain file when you call cmake to generate:
```bash
cmake -B ./build -S ./ "-DCMAKE_TOOLCHAIN_FILE=the_path_to/vcpkg/scripts/buildsystems/vcpkg.cmake"

# e.g
cmake -B ./build -S ./ "-DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake"
```
This command will run vcpkg install for you if you have a `vcpkg.json` file.
### Building on the command line 
After configuration and generation is done, you can run 
```bash
cd build
cmake build .
```
This will build the project.

### Using Visual Studio
On Windows, if you prefer working in Visual Studio, after generation is done, you can open the generated `sln` file and build any target you want.
Just make sure it is the selected as a Startup project. The working directory and post build commands have been automatically configured.

## Useful resources
* [`vulkan-tutorial.com`](https://vulkan-tutorial.com/)
* [`Integrating Dear ImGui in a custom Vulkan renderer - Frguthmann's blog`](https://frguthmann.github.io/posts/vulkan_imgui/)
* [`Walnut by TheCherno`](https://github.com/TheCherno/Walnut) 
* [`thomasherzog response to What things in the Vulkan code should be changed for the ImGui window rendering?`](https://github.com/ocornut/imgui/issues/4115#top)
* [`ImGui GLFW Vulkan example code`](https://github.com/ocornut/imgui/blob/master/examples/example_glfw_vulkan/main.cpp)
* [`Sascha Willems imgui example`](https://github.com/SaschaWillems/Vulkan/blob/master/examples/imgui/main.cpp)
* [`vkguide-implementing imgui`](https://vkguide.dev/docs/extra-chapter/implementing_imgui/) 

