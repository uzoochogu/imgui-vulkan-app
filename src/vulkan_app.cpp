#define GLFW_INCLUDE_VULKAN // GLFW will include Vulkan and do some checks
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES // To help with alignments
                                           // requirements
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // exposes functions that can be used to generate MVP transformations.

#define STB_IMAGE_IMPLEMENTATION // needed to include definitions (function
                                 // bodies)
#include <stb_image.h>           // for loading images, only defines prototypes

#include <algorithm> //necessary for std::clamp
#include <array>
#include <chrono>
#include <cstdint> // for std::uint32_t
#include <cstdlib>
#include <cstring> //may be needed for cstring
#include <fstream>
#include <iostream>
#include <limits> //Necessary for std::numeric_limits
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Vertex data

// To be used in the vertex shader
// GLM provides C++ types that exactly match vector types used
// in shader language
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  // A vertex binding describes at which rate to load data from memory
  // throughout the vertices. It specifies the number of bytes between data
  // entries and whether to move to the next data entry after each vertex or
  // after each instance.
  static VkVertexInputBindingDescription getBindingDescription() {
    // All of our per-vertex data is packaed together in one array, so we're
    // only going to have one binding.
    VkVertexInputBindingDescription bindingDescription{};

    // This specifies the index of the binding in the array of bindings
    bindingDescription.binding = 0;
    // This specifies the number of bytes from one entry to the next
    bindingDescription.stride = sizeof(Vertex);
    // This determines how you mobe to the next data, either move after each
    // vertex or after each instance. We are moving after each instance.
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  // An attribute description struct describes how to extract a vertex attribute
  // from a chunk of vertex data originating from a binding description. We have
  // two attributes, position and color.
  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    // Tells Vulkan from which binding the per-vertex data comes.
    attributeDescriptions[0].binding = 0;

    // This references the location directive of the input in the vertex shader.
    // e.g layout(location = 0) in vec2 inPosition;
    // 0 is the position and has two 32-bit float components.
    attributeDescriptions[0].location = 0;

    // Describes the type of data for the attribute.
    // Albeit confusingly, they are specified using the same enumeration as
    // color format. e.g. float: VK_FORMAT_R32_SFLOAT. use the format where the
    // amount of color channels matches the number of components in the shader
    // data type. Below is for a vec2.
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;

    // format implicitly defines the byte size of attribute data and the offset
    // parameter specifies  the number of bytes since the start of the
    // per-vertex data to read from. The offset macro calculates the offest
    // based on the fact that the binding is loading one Vertex at a time and
    // the position attribute (pos) is at an offset of 0 bytes from the
    // beginning of this struct.
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    // color attribute is described similarly
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    return attributeDescriptions;
  }
};

// Uniform buffer object  descriptor
// GLM data types exactly matches the defintion in the shader
// This is good for bianry compatibilty and operations like
// memcpy a UBO to a VkBuffer.
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

// Array of vertex data. The position and color values are combined into
// one array of vertices. (Interleaving vertex attributes)
// RGB Triangle
/* const std::vector<Vertex> vertices =  {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
}; */
// White, Green and Blue vertices
/* const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
}; */

// Rectangle with color - RGBW
const std::vector<Vertex> vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                      {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                      {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                      {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};

// index buffer
// std::uint16_t for less than 65535 vertices
const std::vector<std::uint16_t> indices = {0, 1, 2, 2, 3, 0};

// Windowing and Vulkan attributes
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Number of frames to be processed concurrently
const int MAX_FRAMES_IN_FLIGHT = 2;

// Loading Shaders
// Reads all bytes from specified file and return a byte array.
static std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename,
                     std::ios::ate |
                         std::ios::binary); // start reading at the end of file.

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file!");
  }

  // determine size of file  using read position
  std::size_t fileSize = static_cast<std::size_t>(file.tellg());
  std::vector<char> buffer(fileSize); // allocate buffer to file size

  // seek to beginning
  file.seekg(0);
  file.read(buffer.data(), fileSize);

  // close file and return bytes
  file.close();
  return buffer;
}

// Provided standard diagnostics validation layer bundled in the Vulkan SDK
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

// list of required device extensions
// VK_KHR_SWAPCHAIN_EXTENSION_NAME is a macro defined as VK_KHR_swapchain
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true; // Runs in debug build
#endif

// Stores properties of window surface for swapchain creation (They must be
// compatible)
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

// stores different Message Queue Indices
struct QueueFamilyIndices {
  std::optional<std::uint32_t> graphicsFamily;
  std::optional<std::uint32_t> presentFamily;

  bool isComplete() {
    // families supporting drawing and presentation may not overlap, we want
    // both to be supported
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

// Create vkDebugUtilsMessengerEXT object
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) { // instance is passed since debug messenger is specific
                         // it and its layers
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else { // return nullptr if it couldn't be loaded
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

// can be static
void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window; // stores window for us.
  VkInstance
      instance; // connection between your application and the Vulkan library
  VkDebugUtilsMessengerEXT debugMessenger; // tell Vulkan about callback,
  VkSurfaceKHR surface; // abstract type of surface to present rendered images
  VkPhysicalDevice physicalDevice =
      VK_NULL_HANDLE;    // stores selected graphics card, implicitly destroyed
                         // along with VKInstance
  VkDevice device;       // stores logical device handle
  VkQueue graphicsQueue; // will store handle to the graphics queue, implicitly
                         // cleaned up on device destruction
  VkQueue presentQueue;  // will store handle to present queue
  VkSwapchainKHR swapChain; // handler for swap chain object
  std::vector<VkImage>
      swapChainImages; // Used to store handles to VkImages in the Swapchain

  VkFormat swapChainImageFormat; // Store chosen format for swap chain images
  VkExtent2D swapChainExtent;    // Store chosen extent for swap chain images
  std::vector<VkImageView>
      swapChainImageViews; // member to store the images views in.
  std::vector<VkFramebuffer>
      swapChainFramebuffers; // member to hold framebuffers for images in
                             // swapchain

  VkRenderPass renderPass; // store the render pass object
  VkDescriptorSetLayout
      descriptorSetLayout; // contains all descriptor bindings.
  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet>
      descriptorSets;              // holds the descriptor set handles
  VkPipelineLayout pipelineLayout; // uniform values for shaders that can be
                                   // changed at drawing time
  VkPipeline graphicsPipeline;     // holds the Graphics Pipeline object

  VkCommandPool commandPool; // Command pools manage memory used to store the
                             // buffer. Command buffers are allocated from them.
  std::vector<VkCommandBuffer> commandBuffers; // To store CommandBuffers

  std::vector<VkSemaphore>
      imageAvailableSemaphores; // semaphore signals image has been acquired
                                // from swapchain, ready for rendering.
  std::vector<VkSemaphore>
      renderFinishedSemaphores; // semaphore signals rendering finished and it
                                // ready for presentation.
  std::vector<VkFence> inFlightFences; // fence to make sure only one frame is
                                       // rendering at a time.

  std::uint32_t currentFrame = 0; // frame index to keep track of current frame.

  bool framebufferResized = false; // true when a resize happens

  VkBuffer vertexBuffer; // vertex buffer handle.
  VkDeviceMemory
      vertexBufferMemory; // store handle of the memory and be allocatable from.

  VkBuffer indexBuffer; // indices need to be uploaded in a GPU accessile buffer
  VkDeviceMemory indexBufferMemory;

  std::vector<VkBuffer>
      uniformBuffers; // holds the multiple uniform buffer for frames in flight
  std::vector<VkDeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  VkImage
      textureImage; // the shader will use this to access pixel values,texels
  VkDeviceMemory textureImageMemory;

  // creating an instance involves specifing some details about the application
  // to driver
  void createInstance() {

    // check validation layers support first:
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "validation layers requested, but not available!");
    }

    VkApplicationInfo
        appInfo{}; // optional but useful to help driver optimize application
                   // (e.g.if it uses well known engine)
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // you specify type of
                                                        // Vulkan struct here
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); // use VK_MAKE_API_VERSION
    appInfo.apiVersion = VK_API_VERSION_1_0;
    // appInfo.pNext  = ??;   //It can point to extension information in the
    // future, ignoring it, value initializes it to nullptr

    VkInstanceCreateInfo createInfo{}; // required, tells driver which global
                                       // x10sion and validation layer to use
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Get required extensions, Specified Enabled extensions
    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // creation information for our debug messenger, placed outside to prevent
    // destruction before instance is created
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

    // global validation layers to enable
    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<std::uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();

      // Added
      populateDebugMessengerCreateInfo(debugCreateInfo);
      createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT
                              *)&debugCreateInfo; // pass info to instance
                                                  // creation infor struct
    } else {
      createInfo.enabledLayerCount = 0;
    }

    // Everything is ready to create an instance, VkResult is result, check for
    // failure
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  bool checkValidationLayerSupport() {
    std::uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(
        &layerCount,
        nullptr); // used just like vkEnumerateInstanceExtensionProperties above

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    // check if all the layer in validationLayers exist in availableLayers list
    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  // Return the required list of extension based on whether validation layers
  // are enabled or not,
  std::vector<const char *> getRequiredExtensions() {
    std::uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(
        glfwExtensions,
        glfwExtensions +
            glfwExtensionCount); // std::vector<T>(InputIt first, InputIt last)

    // conditionally add the debug messenger extension
    // if you have this Error: VK_ERROR_EXTENSION_NOT_PRESENT, some validation
    // layer is missing
    if (enableValidationLayers) {
      extensions.push_back(
          VK_EXT_DEBUG_UTILS_EXTENSION_NAME); // equal to VK_EXT_debug_utils
                                              // string literal , this avoids
                                              // typos
    }

    return extensions;
  }

  // Debug callback function
  // VKAPI_ATTR and VKAPI_CALL ensure function has the right signature for
  // Vulkan to call it has the PFN_vkDebugUtilsMessengerCallbackEXT prototype
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
  }

  void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT; // All except message
                                                       // severity verbose
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT; // enabled all here
    createInfo.pfnUserCallback =
        debugCallback;              // pointer to user defined callback function
    createInfo.pUserData = nullptr; // Optional
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;

    // fill structure with details about the messenger and its callback
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    // Call function to function to create extensionobject if available
    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
                                     &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  // helper function for pickPhysicalDevice()
  // Checks if passed device is suitable for the operation we want to perform
  bool isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    // Check for swapChain Adequacy
    bool swapChainAdequate = false;
    if (extensionsSupported) { // only query for swap chain support after
                               // verifying the extension is available
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      // for now we want at least one supported image format and one supported
      // presentation mode for the given surface
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    std::uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr); // fills the extensionCount

    // All vector to store all available extensions
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    // Copy required extensions defined globally
    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());
    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(
          extension.extensionName); // erase will succeed if present
    }

    return requiredExtensions.empty(); // if empty, extension is present
  }
  // Select a graphics card that supports the features we need. Select only one
  // for now
  void pickPhysicalDevice() {
    // Listing the graphics cards is very similar to listing extensions and
    // starts with querying just the number.
    std::uint32_t deviceCount{0};
    vkEnumeratePhysicalDevices(instance, &deviceCount,
                               nullptr); // stores the number of extension here

    if (deviceCount ==
        0) { // 0 devices with Vulkan support no need for going further.
      throw std::runtime_error("failed to find GPUs with Vulkan support");
    }

    // allocate array to hold alll VkPhysicalDevice handles
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Check for Device suitability
    for (const auto &device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    // Logic to find graphics queue family
    QueueFamilyIndices indices;

    // Extract Queue Families Properties, similar pattern to  Physical Device
    // and Extensions
    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    // Assign index to queue families that could be found

    int i{0};
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueFlags &
          VK_QUEUE_GRAPHICS_BIT) { // support drawing commands
        indices.graphicsFamily = i;
      }

      // Look for queue family that can present to our window surface.
      VkBool32 presentSupport = false;
      // takes in physical device, queue family index and surface
      vkGetPhysicalDeviceSurfaceSupportKHR(
          device, i, surface,
          &presentSupport); // populates presentSupport if yes.
      if (presentSupport) {
        indices.presentFamily = i;
      }

      // break early
      if (indices.isComplete()) {
        break;
      }
      i++;
    }

    return indices;
  }

  void createLogicalDevice() {
    // Get Queue Family index
    QueueFamilyIndices indices = findQueueFamilies(
        physicalDevice); // pass in handle to the current physical device

    // CreateInfos for creating queues from the Queue families
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    // using a set of all unique queue families necessary for required queue
    // value certain due to check in pickPhysicalDevice()
    std::set<std::uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(), indices.presentFamily.value()};

    // populate Device Queue Create info
    float queuePriority = 1.0f;
    for (std::uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;

      // Assign priority for scheduling command buffer execution - Required
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // Specify used device features
    VkPhysicalDeviceFeatures
        deviceFeatures{}; // we don't need anything special, so just define for
                          // now.

    // Create the logical device
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    // Add pointers to the queue creation info and device features structs
    createInfo.queueCreateInfoCount =
        static_cast<std::uint32_t>(queueCreateInfos.size());
    ;
    createInfo.pQueueCreateInfos =
        queueCreateInfos.data(); // points to all createInfos
    createInfo.pEnabledFeatures = &deviceFeatures;

    // The rest of info is like VkInstanceCreateInfo but device specific
    createInfo.enabledExtensionCount =
        static_cast<std::uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      // ignored by current Vulkan implementation (Uses Instance specific
      // Vlaidation). Include for compatiblity reasons
      createInfo.enabledLayerCount =
          static_cast<std::uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    // No need for device specific extensions for now

    // Instantiate logical device
    // physical device to inteface with, queue and usage info, optional
    // allocator callback, pointer to variable to store the logical device
    // handle in.
    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    // Retrieves queue handles for each queue family.
    // Passed logical device, queue family, queue index (0 in this case, since
    // we are only creating one), pointer to variable to store queue handle in
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);

    // If the queue families are the same, then we only need to pass its index
    // once(both handles will have the same value) call to retrieve the queue
    // handle:
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
  }

  // Used to create the VkSurface_KHR instance
  void createSurface() {
    // Takes simple paratmeters instead of a struct
    // VkInstance, GLFW window pointer, custom allocator and pointer to
    // VksurfaceKHR
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  // Settings for Swap Chain

  // Populates SwapChainSupportDetails struct
  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details;

    // Basic surface capabilities: returned into a VkSurfaceCapabilitiesKHR
    // struct
    //  Takes in specified VkPhysicalDevice adn VkSurfaceKHR window surface
    //  intom account.
    // These two will be in all support querying functions as they are core
    // components of the swap chain.
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                              &details.capabilities);

    // Query for surported surfacr formats. List of structs. So 2 function calls
    std::uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         nullptr);

    if (formatCount != 0) {
      details.formats.resize(formatCount); // make sure the vector is resized to
                                           // all available formats.
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                           details.formats.data());
    }

    // Query the presentation modes (Same procedure)
    std::uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                              &presentModeCount, nullptr);

    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, surface, &presentModeCount, details.presentModes.data());
    }
    return details;
  }

  // Surface format
  // Pass the formats member of SwapChainSupportDetails
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // Each entry in vector contains format and a colourSpace member.
    // format specifies the color channels and types, e.g
    // VK_FORMAT_B8G8R8A8_SRGB stores B G R A channels i 8Bit (32Bits per pixel)
    // colorSpace indicates if SRGB colour space is supported using the
    // VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag. or
    // VK_COLORSPACE_SRGB_NONLINEAR_KHR in old versions of the spec

    // Go through list and see if preferred combination is available
    // VK_FORMAT_B8G8R8A8_SRGB because SRGB results in better conceived colours
    // and is a common standard
    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    // Just choose the first if the most preferred is not availble
    return availableFormats[0];
  }

  // Presentation Mode
  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes) {

    for (const auto &availablePresentMode : availablePresentModes) {
      // Nice trade-off if energy usage is mot a concern. Minimizes tearing
      // while maintaining low latency Up to date New images are rendered until
      // the vertical blank Doesn't block application when queue is full,
      // replace old images in queue with newer ones.
      if (availablePresentMode ==
          VK_PRESENT_MODE_MAILBOX_KHR) { // aka Tripple Buffering
        return availablePresentMode;
      }
    }

    // Guaranteed to be available
    // Swap chain is a queue, FIFO is observed. Similar to Vertical sync. After
    // refresh is called Vertal blank
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  // Swap Extent
  // VkSurfaceCapabilitiesKHR contains the range of possible resolution of
  // window in pixels We can't use WIDTH and HEIGHT declared because it can
  // vary. Also, that is in screen coordinates unit (not pixels)
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    // Use currentExtent member of VkSurfaceCapabilitiesKHR, unless
    // Some window managers set the currentExtent to maximum value of uint32_t
    // as a special case(else block)
    if (capabilities.currentExtent.width !=
        std::numeric_limits<std::uint32_t>::max()) {
      return capabilities.currentExtent;
    } else { // Pick resolution that best matches window within minImageExtent
             // and maxImageExtent in pixels
      int width{0}, height{0};
      // Returns the current height of the window( window's frame buffer) in
      // pixels
      glfwGetFramebufferSize(window, &width, &height);

      // Create actualExtent struct with the size from glfw window in pixels
      VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

      // Clamps value to used to bound the values between the allowed min and
      // max supported by the implementation returns passed value unless it is
      // higher or lower than the bounds, then it returns the closest bound.
      actualExtent.width =
          std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                     capabilities.maxImageExtent.width);
      actualExtent.height =
          std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                     capabilities.maxImageExtent.height);

      return actualExtent;
    }
  }

  void createSwapChain() {
    // Get Swapchain suport info
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(
        physicalDevice); // pass in physical device handler

    // Choose surface formats
    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);

    // Choose presentation mode
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);

    // Get Screen Extents (in pixels)
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // Decide how many images we would like to have in the swap chain
    // Impleemntation specifies the minimum number it requires to fucntion
    // To avoid waiting on driver to complete internal operations before
    // acquiring another images to render request at least one more image than
    // minimum
    std::uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    // Don't exceed maximum number of images
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    // fill in the creation structure
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface; // pass in the surface handler

    // specify details of the swap chain images
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;

    // imageArrayLayers specifies the amount of layers each images consists of
    // Always 1 unless you are developing steroscopic 3D application
    createInfo.imageArrayLayers = 1;

    // imageUsage Specifies what kind of operations we use the images in the
    // swap chain for. For now we render directly to them i.e.  they are used as
    // color attachment You can render to a separate image first to perform
    // operation like post processing In that case use
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT and then use a memory op to transfer the
    // rendered image to a swap chain image
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // Specify how to handle swap chain images used across multiple queue
    // families.
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    std::uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                          indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      // graphics queue family is different from the presentation queue
      // Drawing on images in the swap chain from the graphics queue and submit
      // to presentation queue
      // Images can be used across multiple queue families without explicit
      // owership transfers
      createInfo.imageSharingMode =
          VK_SHARING_MODE_CONCURRENT; // doing this here for ease, later we can
                                      // ownership control

      // Concurrent mode requires you specify in advancem between with queue
      // families ownership will be shared
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      // image owned by one queue family at a time,must be transfered before
      // using it in another family.
      createInfo.imageSharingMode =
          VK_SHARING_MODE_EXCLUSIVE;            // Offers best performance
      createInfo.queueFamilyIndexCount = 0;     // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    // Specify that a certain transform should be applied to images in the swap
    // chain if it is supported by setting to capabilities.supportedTransforms,
    // otherwise specify current transformation
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

    // Specifies if the Alpha channel should be used for blending with other
    // windows in the window system. We ignore it here.
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode; // pass the presentation mode handler
    createInfo.clipped =
        VK_TRUE; // TRUE means we don't care about the color of obscured pixels.
                 // Use FALSE unless it is really needed

    // Assume we only ever create one swap chain
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    // Create the swapChain
    // Pass in device, creation info, custom allocator (nullptr) for now, and
    // the handler to store it in
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create swap chain!");
    }

    // Retrieve the swap chain images
    // use the count, resize pattern
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                            swapChainImages.data()); // store the images

    // Store chosen Format and Extent
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  void createImageViews() {
    swapChainImageViews.resize(
        swapChainImages
            .size()); // resize to fit all of the image views we'll be creating

    // Iterate over all of the swap chain images
    for (std::size_t i = 0; i < swapChainImages.size(); i++) {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];

      // Specify how the image date should be interpreted.
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat;

      // The component field allow you to swizzle the color channels around.
      // e.g map all the channels to red channel for a monochrome texture or
      // map constant values of 0 and 1 to a channel
      createInfo.components.r =
          VK_COMPONENT_SWIZZLE_IDENTITY; // stick to default mapping
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

      // Define the image's purpose and part of image should be accessed
      // Use images as color targets and no mipmapping levels or multiple layers
      createInfo.subresourceRange.aspectMask =
          VK_IMAGE_ASPECT_COLOR_BIT; // used a color target
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      // Call creation function (each should be explicitly destroyed)
      if (vkCreateImageView(device, &createInfo, nullptr,
                            &swapChainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image views!");
      }
    }
  }

  void createRenderPass() {
    // we'll just have a single color buffer attachment
    // represented by one of the images of the swap chain
    VkAttachmentDescription colorAttachment{};
    // format of color attachment should match the format of the
    // swap chain images.
    colorAttachment.format = swapChainImageFormat;
    // stick to 1 sample, no multisampling yet
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

    // loadOp and storeOp determine what to do with the data in the attachment
    // before rendering and after rendering.
    // For Don't care, existing contents are undefined, clear clears the value
    // to a constant. load preserves existing contents we want to use the clear
    // operation to clear the framebuffer to black before drawing a new frame.
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    // Rendered content will be stored in memory and can be read later.
    // We are interested in seeing the rendered triangle on the screen
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    // loadOp and storeOp apply to color and depth data and
    // stencilStoreOp apply to stencil data
    // We are not doing anything with the stencil buffer so results of loading
    // and storing are irrelevant
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    // Textures and framebuffers in Vulkan are represented by VkImage objects
    // with a certain pixel format, however the layout of the pixels in memory
    // can change based on what you're trying to do with an image.

    // Layout the image will have before the render pass begins.
    // Undefined means we don't care what previous layout the image was in,
    // we don't care since we are clearing it.
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // The layout to automatically transition to when the render pass finishes.
    // final layout should be Images to be present in the swap chain,
    // we want the image to be ready for presentation using the swapchain after
    // rendering.
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Subpasses and attachment references
    VkAttachmentReference colorAttachmentRef{};
    // this param specifies which attachment to reference by its index
    // in the attachment descriptions array. Our array is only a single
    // VkAttachmentDescription so 0 index.
    colorAttachmentRef.attachment = 0;
    // layout specifies which layout we would like the attachment to have
    // during a subpass that uses this reference.
    // We are using the attachment to function as a color buffer so this is the
    // best
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // subpass
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

    // specify the reference to the color attachment.
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    // Adding subpass dependencies
    VkSubpassDependency dependency{};
    // These specify the indices of the dependency and the dependent subpass.
    // This refers to the implicit subpass before or aftewr the render pass
    // depending on whether it is specified in srcSubPass or dstSubPass
    // 0 refers to our subpass (first and only one). dst should be higher than
    // src to prevent cycles in dependency graph (Unless one is
    // SUBPASS_EXTERNAL)
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;

    // Specify the operations to wait on and stage in which operations occur
    // Wait on swap chain to finish reading fromm image before we can access it.
    // so wait on the color attachment output stage itself.
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;

    // The operations that should wait on this are in the coolor attachment
    // stage and involve the writing of the color attachment. These prevent the
    // transition from happening until it's actually necessary or allowed: when
    // we want to start writing colors to it
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // Create the render pass object
    // fill the creation struct with an array of attachments and subpasses.
    // attachments are referenced using the indices of this array by the
    // VkAttachmentReference object
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    // pass the array of dependencies here
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createDescriptorSetLayout() {
    // Every binding needs to be described through this struct.
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    // binding used int he shader
    uboLayoutBinding.binding = 0;
    // type of descriptor is a Uniform buffer object
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Number of values in the array
    // Can be used the specify a transformation for each bones in a skeleton for
    // skeletal animation for example. Our MVP transformation is in a single
    // uniform buffer object so 1 is specified.
    uboLayoutBinding.descriptorCount = 1;
    // the shader stage it is going to be referenced. It can be a combination of
    // VkShaderStageFlagBits values or VK_SHADER_STAGE_ALL_GRAPHICS
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    // only relevant for image smaple related descritors.
    uboLayoutBinding.pImmutableSamplers = nullptr; // optional

    // Create VkDescriptorSetLayout
    // VkDescriptorSetLayoutCreateInfo
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &descriptorSetLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
    }
  }

  // Creates graphics pipeline, loads shaders code
  void createGraphicsPipeline() {
    // load bytecode of the two shaders:
    auto vertShaderCode =
        readFile("../../shaders/vert.spv"); // path relative to the executable
    auto fragShaderCode = readFile("../../shaders/frag.spv");

    std::cout << "Check for correct load\n";
    std::cout << "Loaded vertex shader bytes: " << vertShaderCode.size()
              << "\n";
    std::cout << "Loaded shader shader bytes: " << fragShaderCode.size()
              << "\n";

    // Load ShaderModules:
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    // Programmable stage
    // Create Pipeline Shader stage for Vertex Shader
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // tells vulkan in which pipeline stage shader will be used.
    // Enums for other programmable stages exist
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    // specify the shader module containing the code
    vertShaderStageInfo.module = vertShaderModule;
    // The shader code entry point
    vertShaderStageInfo.pName = "main";

    // Create Pipeline Shader stage for Fragment Shader
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    // Array to contain two structs
    // Would be referenced in the pipeline creation
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    // Get vertex data
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    // Fixed function stage
    // Vertex input
    // We are filling this struct to specify that there is no vertex data to
    // load for now
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    // Set up graphics pipeline to accept vertex data
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // Input Assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // For more flexibility, define viewport and scissor rectangles as a dynamic
    // state in command buffer
    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                 VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount =
        static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    // Specify their count at pipeline  creation time
    // The actual viewport and scissor rectangle will be later set up at drawing
    // time
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterizer Creation Struct
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // If true, then fragments beyond the near and far planes are clamped as
    // opposed to discarding them This is useful in some special cases like
    // shadow maps You must enable a GPU feature to use this
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable =
        VK_FALSE; // set to true to disable output to frame buffer

    // polygonMode determines how fragments are generated for geometry.
    rasterizer.polygonMode =
        VK_POLYGON_MODE_FILL; // fill the area of the polygon with fragments
    rasterizer.lineWidth =
        1.0f; // thickness of lines in terms of number of fragments.

    rasterizer.cullMode =
        VK_CULL_MODE_BACK_BIT; // type of face culling, cull the back faces
    rasterizer.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE; // vertex order to determine
                                         // front-facing

    // The rasterizer can alter depth values by adding a constant value or
    // biasing them based on a fragment's slope. This is sometimes used for
    // shadow mapping
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // optional
    rasterizer.depthBiasClamp = 0.0f;          // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f;    // Optional

    // Multisampling Creation Struct
    // Disabled for now
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;          // Optional
    multisampling.pSampleMask = nullptr;            // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE;      // Optional

    // color blending
    // configuration per attached framebuffer
    // Allows configuring this method of color blending: Mix the old and new
    // value to produce a final color
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable =
        VK_FALSE; // new color from frag shader passes through unmodified.
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;             // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;  // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;             // Optional

    // global color blending settings
    // If enabled, allows blending through bitwise combination, (but disables
    // per framebuffer config)
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;   // unmodified
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // Tell Vulkan which descriptors the shaders will be using
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    // Push constants are another way of passing dynamic values to shaders
    pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    // Create pipelineLayout object
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    // Creating the Graphics Pipeline
    //  Reference VkPipelineShaderStageCreateInfo structs
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    // Then we reference all the structures describing fixed function stage
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr; // optional
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    // Then the pipeline layout, a Vulkan handle and not a struct pointer
    pipelineInfo.layout = pipelineLayout;
    // Then the render pass and index of the subpass where grapphics pipeline is
    // used We can use other render passes in this pipeline but it must be
    // compatible with renderPass
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    // We are not deriving from another Graphics pipeline.
    // We are using a single pipeline
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1;              // Optional

    // Create Grpahics Pipeline
    // Note that the function is designed to take multiple
    // VkGraphicsPipelineCreateInfo objects anbd create multiple VkPipeline
    // object in a single call The second parameter was VK_NULL_HANDLE argument,
    // references an optional VkPipelineCache object to help reuse data across
    // mutliple calls of the function
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    // Shader Modules clean up
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
  }

  // Create VkShaderModule to wrap code passed into the graphics pipeline
  VkShaderModule createShaderModule(const std::vector<char> &code) {
    // specify pointet to the buffer witht he bytecode and length
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    // pCode is a uint32_t pointer
    // reinterpret cast, data alignment assured for std::vector
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    // create ShaderModule
    VkShaderModule shaderModule; // only needed for pipeline creation
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
  }

  // Creates framebuffers per image in swapchain
  void createFramebuffers() {
    // resize to hold all
    swapChainFramebuffers.resize(swapChainImageViews.size());

    // iterate through image views, create framebuffers from them
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      VkImageView attachments[] = {swapChainImageViews[i]};

      // Create Framebuffer
      VkFramebufferCreateInfo framebufferInfo{};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      // specify with which renderPass, the framebuffer needs to be compatible
      // with. render pass must be compatible with framebuffer usually by having
      // the same number and type of attachments.
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      // number of layers in image arrays, our swapchain images are single so 1
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  // Command pools manage the memory that is used to store the buffers and
  // command buffers are allocated from them. We have to create a command pool
  // before we can create command buffers.
  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // There are 2 possible flags for command pools
    // This flag Allow command buffers to be rerecorded individually, without
    // this, they would all have be reset together
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // Each command pool can only allocate command buffers that are submitted on
    // a single type of queue. We're going to record commands for drawing, which
    // is why we've chosen the graphics queue family.
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    // Create CommandPool
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  // Abstract image creation, handles image object creation and memory
  // allocation.
  void createImage(std::uint32_t width, std::uint32_t height, VkFormat format,
                   VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    // kind of coordinate system the texels in the image are going to be
    // addressed 1D can store array of data or gradient, 2D are mainly used for
    // textures and 3D are used to store voxel volumes.
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    // specifies the dimensions of the image, texels in each axis. thus depth is
    // 1 not 0.
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    // Our texture will not be an array and we won't be using mipmapping for
    // now.
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    // Not usable by the GPU, the very first transition will discard the texels.
    // we're first going to transition the image to be a transfer destination
    // and then copy texel data to it from a buffer object, so we don't need
    // this property
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    // The image will only be used by one queue family: the one that supports
    // graphics (and therefore also) transfer operations.
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // relates to multisampling. Only relevant for images used as attachments,
    // so stick to 1 sample.
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = 0; // Optional

    if (vkCreateImage(device, &imageInfo, nullptr, &textureImage) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create image!");
    }

    // Similar to allocating memory for a buffer.
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, textureImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate image memory");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
  }

  // We'll load an image and upload it into a Vulkan image object, using command
  // buffers.
  void createTextureImage() {
    int texWidth, texHeight, texChannels;
    // current image is 1920×1440
    // path, out variables for width height, actual channels channels and then
    // the number of cahnnels to load STBI_rgb_alpha forces loading with alpha
    // channel even if it is not present.
    // Pointer returned is the first element in the array of pixel values. The
    // pixels are laid out row by row with 4 bytes per pixel in the case of
    // STBI_rgb_alpha for a total of texWidth * texHeight * 4 values
    stbi_uc *pixels = stbi_load("../../textures/texture.jpg", &texWidth,
                                &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    // HOST Visible memory so we can map it and usable as a transfer source so
    // it can be later copied to an image
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // copy the pixel values
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    // clang-format off
    // Reason: Show Hierarchy
     memcpy(data, pixels, static_cast<std::size_t>(imageSize));
    // clang-format on
    vkUnmapMemory(device, stagingBufferMemory);

    stbi_image_free(pixels); // frees image array

    // Format - use same format as the pixels in the buffer, else copy operation
    // will fail. Tiling - We choose for texels are laid out in an implentation
    // defined order (TILING_OPTIMAL) for optimal access and not row-major in
    // our pixel array. (TILING_LINEAR) since we are using a staging buffer we
    // can use optimal access. Usage - image is going to be used as destination
    // for the buffer copy. We would also access the image from the shader to
    // color our mesh properties -
    createImage(
        texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    // Copy staging buffer to the texture image
    // Transition the texture image
    // Specify the undefined since it was created with that and we don't care
    // about its contents before the copy operation.
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8_SRGB,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // Execute the buffer to image copy
    copyBufferToImage(stagingBuffer, textureImage,
                      static_cast<std::uint32_t>(texWidth),
                      static_cast<std::uint32_t>(texHeight));

    // one more transition to prepare for shader access, to be able to start
    // sampling from the texture image in the shader
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8_SRGB,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  // Changes images to the right layout, used when we want to make sure image is
  // in the right layout for the vkCmdCopyBufferToImage command
  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // We would perform layout transitions using pipeline barriers. Which syncs
    // access to resources. Image memeory barrier can be used for images, There
    // is an equivalent  buffer memory barrier for buffers. It can transition
    // image layouts and transfer queue family ownership when
    // VK_SHARING_MODE_EXCLUSIVE is used.
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // can be VK_IMAGE_LAYOUT_UNDEFINED if you don't care about the existing
    // contents of the image.
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    // We are not using barrier to transfer queue family ownership. Must be set
    // to this and not default value.
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    // specify the image that is affected and the specific part of the image.
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    // our image is not an array and does not have mipmapping levels so only one
    // level and layer are specified.
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    // Handling Undefined -> Transfer destination
    // transfer destination -> shader reading
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      // 0 means we are not waiting for anything
      barrier.srcAccessMask = 0;
      // transfer writes must occur in the pipeline transfer stage
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

      // we specify the earliest possible pipeline stage
      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      // Pseudo stage where transfers happen
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      // image written in same pipeline stage
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      // allow shader reading access
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      // subsequent read by the frag shader
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }
    // The first parameter after the command buffer specifies in which pipeline
    // stage the operations occur that should happen before the barrier. The
    // second parameter specifies the pipeline stage in which operations will
    // wait on the barrier. The pipeline stages that you are allowed to specify
    // before and after the barrier depend on how you use the resource before
    // and after the barrier.
    // The third parameter is either 0 or VK_DEPENDENCY_BY_REGION_BIT. The
    // latter turns the barrier into a per-region condition. The last three
    // pairs of parameters reference arrays of pipeline barriers of the three
    // available types: memory barriers, buffer memory barriers, and image
    // memory barriers like the one we're using here.
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(VkBuffer buffer, VkImage image, std::uint32_t width,
                         std::uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    // specify which part of the buffer is going to be copied to which part of
    // the image
    VkBufferImageCopy region{};
    // specifies the buffer offset at which the pixel values start
    region.bufferOffset = 0;
    // Specifies how the pixels are laid out in memory, 0 for both leads to
    // tightly packed pixels. Else you can have some padding bytes.
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    // Below indicates to which parts we want to copy the pixels
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    // Enqueue buffer to image copy operations
    // specified src and dst
    // 4th param indicates which layout the image is currently using. We would
    // assume that the image is already transitioned to layout optimal for
    // copying pixels to.
    // We are only transfering 1 chunk of pixels to the whole image, an array of
    // VkBufferImageCopy can be specified to perform many different copies from
    // this buffer to the image in one operation.
    vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
  }

  std::uint32_t findMemoryType(std::uint32_t typeFilter,
                               VkMemoryPropertyFlags properties) {
    // Query info about the available types of memory
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    // Find a memory type that is suitable for the vertex buffer
    for (std::uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      // The typeFilter parameter will be used to specify the bit field of
      // memory types that are suitable. We  also need to be able to write our
      // vertex data to that memory We may have more than one desirable
      // property, so we should check for equality with the desired properties
      // bit field.
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find suitable memory type!");
  }

  // Handles all buffer creation operations
  // buffer size, memory properties and usage are customizable.
  // buffer and bufferMemory are out variables
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) {
    // fill up VkBufferCreateInfo struct
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // specifies the size of the buffer in bytes.
    bufferInfo.size = size;
    // usage indicates the use case of the data in the buffer
    // Different purposes can be bitwised ORed.
    bufferInfo.usage = usage;
    // Just like swap chain images buffers can be owned by a specific queue
    // family or shared between multiple at the same time. Here it is only used
    // from the graphics queue so stick to exclusive.
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // Used to Configure sparse buffer memory (not relevant now)
    bufferInfo.flags = 0;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create vertex buffer!");
    }
    // Memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    // Memory allocation
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate vertex buffer memory!");
    }

    // Once successful, associate this memory with the buffer
    // The fourth param is the offset within the region of memory.
    // If offset is non-zero, it is required to be divisible by
    // memRequirements.alignment
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  // helper functions of copyBuffer
  VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    // start recording the command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // We're only going to use the command buffer once and wait
    // with returning from the function until the copy operation
    // has finished executing. It is good practice to communicate this
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);
    // Execute the command buffer
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    // clean up the command buffer used for the transfer operation
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }

  // Would be used to copy data between buffer. e.g. staging buffer in
  // CPU accessible memory to  vertex buffer, or a VkBuffer (staging resource)
  // to an image
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    // This function transfers the contents of tyhe buffer.
    // It also takes the array of regions to copy.
    // The regions are defined in VkBufferCopy struct. It is not possible to
    // specify VK_WHOLE_SIZE here.
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  void createVertexBuffer() {
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    // Temporary Buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    // usage is as a transfer src bit
    // Memory property flag VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT indicates that
    // we are able to map it so we can write to it from the CPU.
    // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ensures that the mapped memory
    // always matches the contents of the allocated memory.
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // Filling the staging buffer
    void *data;
    // allows us to access a region of specified memory resource defined by an
    // offset and size here 0, bufferSize. Use VK_WHOLE_SIZE to map to all the
    // memory. No flags were passed is the 5th param (currently not supported,
    // must be 0). Last param specifies output for the pointer to mapped memory
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(),
           static_cast<size_t>(bufferSize)); // Now memcpy
    vkUnmapMemory(device, stagingBufferMemory);

    // usage is as a destination buffer for a transfer and a vertex buffer
    // Memory property flag VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT indicates that
    // the memory is device local. We can't use vkMapMemory  but can copy to it
    // from a staging buffer.
    createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    // Clean up staging memory
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  // Similar to createVertexBuffer
  void createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    // Temporary Buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    // Filling the staging buffer
    void *data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), static_cast<size_t>(bufferSize)); // Now memcpy
    vkUnmapMemory(device, stagingBufferMemory);

    // usage is as a destination buffer for a transfer and an index buffer
    createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    // Clean up staging memory
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
  }

  void createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    // level specifies if the allocated command buffers are primary or secondary
    // command buffers primary can be submitted to a queue for execution but
    // can't be called from other command buffers. secondary can't be submitted
    // but can be called from primary command buffers. secondary is helpful to
    // reuse common operations from primary command buffers.
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    // number of buffers to allocate
    allocInfo.commandBufferCount =
        static_cast<std::uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }
  }

  // Writes commands we want to execute into a command buffer.
  // We take in the VkCommandBuffer and the index of the current swapchain image
  // we want to write to
  void recordCommandBuffer(VkCommandBuffer commandBuffer,
                           std::uint32_t imageIndex) {
    // Call vkBeginCommandBuffer first

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    // pInheritanceInfo is only relevant for secondary command buffers.
    // It specifies which state to inherit from the calling primary command
    // buffers.
    beginInfo.pInheritanceInfo = nullptr; // Optional

    // Note that a if the command buffer was already recorded once, then another
    // call will implicitly reset it. It is not possible to append commands at a
    // later time.
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    // pass in previously created renderPass
    renderPassInfo.renderPass = renderPass;
    // attachments to bind
    // bind framebuffer for the swapchain image we want to draw to.
    // Using imageIndex parameter, we can pick the right framebuffer for the
    // current swapchain image. Remember We created a framebuffer for each swap
    // chain image where it is specified as a color attachment.
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];

    // These 2 define the size of the render area.
    //  The render area defines where shader loads and stores will take place
    //  Pixels outside this region will have undefined values.
    //  It should match size of attachments for best performance.
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent;

    // clear color set to be black with 100% opacity
    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    // clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR which we used
    // as load operation for the color attachment.
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    // Begin Render pass
    // All functions that record commands have a vkCmd prefix
    // They all return void, so no error handling until we've finished
    // recording. First arg to a command is always the command buffer to record
    // the channel to. Second param specifies the details of the render pass.
    // Final parameter controls how the drawing commands within the render pass
    // will be provided. in CONTENTS_INLINE commands will be embedded in the
    // primary command buffer itself, no secondary buffers will be executed
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    // clang-format off
    // Reason : Hierarchy
      // Bind graphics pipeline
      // second param specifies if the pipeline object is graphics or compute
      // pipline. Then we pass in the graphics pipeline (contians whic operations
      // to execute and attachment to use)
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        graphicsPipeline);

      VkBuffer vertexBuffers[] = {vertexBuffer};
      VkDeviceSize offsets[] = {0};
      // Binds vertex buffers to bindings
      // second param is the offset, third param is the number of bindings we're
      // going to specify vertex buffer for.
      // last two are the array of vertex buffers to bind and byte offsets to
      // start reading from.
      vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

      // Binds index buffers
      // Takes in index buffer, byte offset, type of index data
      vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

      // Specify viewport and scissor state, since they were set to be dynamic
      VkViewport viewport{};
      viewport.x = 0.0f;
      viewport.y = 0.0f;
      viewport.width = static_cast<float>(swapChainExtent.width);
      viewport.height = static_cast<float>(swapChainExtent.height);
      viewport.minDepth = 0.0f;
      viewport.maxDepth = 1.0f;
      vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

      VkRect2D scissor{};
      scissor.offset = {0, 0};
      scissor.extent = swapChainExtent;
      vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

      // bind the right descriptor set for each frame to the descriptors in the
      // shader before draw call
      // Specifiy binding to graphics pipeline, then the layout the descriptors
      // are based on, Index of first descriptor sets, the number of sets to bind
      // and the array of the sets to bind. last 2 params are used for dynamic
      // descriptors (not used here).
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelineLayout, 0, 1, &descriptorSets[currentFrame],
                              0, nullptr);

      // Draw command
      // Params:
      // Command Buffer
      // vertexCount: size of vertices contains this info
      // instanceCount: Used for instanced rendering, use 1 if you're not doing
      // that. firstVertex: Used as an offset into the vertex buffer, defines the
      // lowest value of gl_VertexIndex. firstInstance: Used as an offset for
      // instanced rendering, defines the lowest value of gl_InstanceIndex.
      // vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0,
      // 0);

      // Similar to vkCmdDraw
      // Params:
      // Command Buffer
      // indexCount - represents the number of vertices that will be passed to the
      // vertex shader instanceCount (1, we are not using instancing), firstIndex
      // - offset into the index buffer (1 means GPU would read from second index)
      // vertexOffset - offset to add to the index buffer
      // firstInstance - offset for instancing (not used)
      vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0,
                      0, 0);
    // clang-format on

    // end render pass
    vkCmdEndRenderPass(commandBuffer);

    // Finish recording the command buffer
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }

  // allocates uniform buffers
  void createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   uniformBuffers[i], uniformBuffersMemory[i]);
      // map buffer right after creation, to get a pointer to which we can write
      // data later on. The buffer strays mapped to this pointer for the
      // application's whole lifetime - Persistent mapping. this is used in all
      // Vulkan implementations.
      vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0,
                  &uniformBuffersMapped[i]);
    }
  }

  // Allocate descriptor sets
  void createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // We allocate one of these descriptors for every frame.
    poolSize.descriptorCount = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    // Also specify the maximum number of descriptor sets that may ne allocated
    poolInfo.maxSets = static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT);
    // determines if individual descriptor sets can be freed or not.
    //  Default value is VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 0
    poolInfo.flags = 0; // Optional

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor pool!");
    }
  }

  void createDescriptorSets() {
    // specify the descriptor pool to allocate from
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    // specify the descriptor pool to allocate from
    allocInfo.descriptorPool = descriptorPool;
    // one descriptor set for each frame in flight, all with the same layout
    allocInfo.descriptorSetCount =
        static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT);
    // descriptor layout to base descriptor sets on
    // we do need all the copies of the layout to match the number of sets.
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    // Allocates descriptor sets, each one with one uniform buffer descrriptor
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    // Configure the descriptors within
    // Descriptors that refer to buffers, like our uniform buffer descriptor,
    // are configured with a VkDescriptorBufferInfo struct. This structure
    // specifies the buffer and the region within it that contains the data for
    // the descriptor.
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      VkDescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = uniformBuffers[i];
      bufferInfo.offset = 0;
      // overwriting the whole buffer, you can also use the VK_WHOLE_SIZE value
      // to overwrite the whole buffer
      bufferInfo.range = sizeof(UniformBufferObject);

      VkWriteDescriptorSet descriptorWrite{};
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      // the descriptor set and binding to update
      descriptorWrite.dstSet = descriptorSets[i];
      descriptorWrite.dstBinding = 0;
      // descriptor can be arrays but not used here.
      descriptorWrite.dstArrayElement = 0;
      // It's possible to update multiple descriptors at once in an array,
      // starting at index dstArrayElement
      descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      // how many array elements you want to update
      descriptorWrite.descriptorCount = 1;

      // These reference an array with descriptorCount structs that actually
      // configure the descriptor depending on the type you actually need to use
      // pBufferInfo is used for descriptors that refer to buffer data.
      // Our descriptor is based on buffers
      descriptorWrite.pBufferInfo = &bufferInfo;
      //  pImageInfo is used for descriptors that refer to image data, and
      //  pTexelBufferView is used for descriptors that refer to buffer views
      descriptorWrite.pImageInfo = nullptr;       // Optional
      descriptorWrite.pTexelBufferView = nullptr; // Optional

      // applies updates,  It accepts two kinds of arrays as parameters: an
      // array of VkWriteDescriptorSet and an array of VkCopyDescriptorSet. The
      // latter can be used to copy descriptors to each other, as its name
      // implies.
      vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }
  }

  // Creates semaphores and fences used in the program
  void createSyncObjects() {
    // resize sync objects
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    // Fill VkSemaphoreCreateInfo struct
    VkSemaphoreCreateInfo semaphoreInfo{};
    // Current version of API doesn't have any req fields other than this
    // flags and pNext fields exist but no functionality for now
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // This is awork around so that for the first time we call vkWaitForFences
    // (e.g. For first frame) it returns immediately, since the fence is already
    // signaled.
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // populate vectors
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // creating the semaphores and fence , similar pattern
      if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &imageAvailableSemaphores[i]) != VK_SUCCESS ||
          vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &renderFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) !=
              VK_SUCCESS) {
        throw std::runtime_error("failed to create synchronization objects - "
                                 "semaphores and fences - for a frame!");
      }
    }
  }

  // Cleans up previos versions of swapchain, image views and framebuffers
  void cleanupSwapChain() {
    for (auto framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    // Destroy views
    for (auto imageView : swapChainImageViews) {
      vkDestroyImageView(device, imageView, nullptr);
    }

    // Destroy swapchain before the device
    vkDestroySwapchainKHR(device, swapChain, nullptr);
  }

  // recreates swap chain to maintain compatibility with the window surface
  void recreateSwapChain() {
    // To handle minimization (frame buffer size = 0)
    // pause until the window is in the foreground again
    int width = 0, height = 0;
    // check for size
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    // Don't touch resources that may still be in use.
    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    createSwapChain();
    // images views are directly based on swap chain images
    createImageViews();
    // framebuffers directly depend on the swap chain images
    createFramebuffers();
  }

  void drawFrame() {
    // Waiting for the previous frame
    // Function takes an array of fences and waits on the host
    // for either any or all the fences to be signaled before returning
    // VK_TRUE passed here means we want to wait for all the fences (We are
    // passing just one). We set timeout to max value of 64 bit uint. (disabling
    // timeout). It waits for inFlightFence to be signaled, which is only
    // signaled when a frame has finished rendering.
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE,
                    UINT64_MAX);

    // Acquiring an image from the swap chain
    std::uint32_t imageIndex;
    // Logical device and swap chain from which we wish to acquire an image.
    // timeout in nanoseconds for an image to become available. (Max value used
    // to effectively disable it) next 2 params are the sync objects that are to
    // be signaled when presentation engine is finished using the image. So we
    // can start drawing to it. We can specify a semaphore, fence or both. last
    // param specifies a variable to output the index of the swapchain image
    // that has become available. imageIndex refers to the swapChainImages
    // array. We'll use this index to pick the VkFrameBuffer.
    VkResult result = vkAcquireNextImageKHR(
        device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame],
        VK_NULL_HANDLE, &imageIndex);

    // recreateSwapChain
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
      // swap chain has become incompatible with the surface and can no longer
      // be used for rendering e.g. after a window resize.
      recreateSwapChain();
      // try again in next draw call
      return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
      // VK_SUBOPTIMAL_KHR means swap chain can still be used to successfully
      // present to the surface, but the surface properties are no longer
      // matched exactly. we are proceeding anyway. Both VK_SUCCESS and
      // VK_SUBOPTIMAL_KHR are considered "success" return codes.
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(currentFrame);

    // Only reset the fence if we are submitting work, this prevents a deadlock
    // if we return early without resetting.
    vkResetFences(device, 1,
                  &inFlightFences[currentFrame]); // Manually reset fence to
                                                  // unsignaled state

    // Recording the command buffer
    // second param is a VkCommandBufferResetFlagBits, 0 for now. Nothing
    // special.
    vkResetCommandBuffer(commandBuffers[currentFrame],
                         0); // to make sure it is able to be recorded.
    recordCommandBuffer(commandBuffers[currentFrame],
                        imageIndex); // record the commands we want

    // Submitting the command buffer
    // Queue submission and synchronization
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // which semaphore to wait on before execution begins
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    // which stage of the pipeline to wait
    // wait with writing colors to the image until it is available so specify
    // stage of graphics pipeline that writes to the color attachment.
    // Theoretically, implementation can already start executing our vertex
    // shader and such when image is not yet available.
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    // each entry in waitStages corresponds to  waitSemaphore
    submitInfo.pWaitDstStageMask = waitStages;
    // command buffer to actually submit for execution
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

    // Semaphores to signal once the command buffer(s) have finished execution.
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // submit the command buffer to the graphics queue.
    // It takes in an array of VkSubmitInfo structures for efficiency at high
    // workloads. Last param is an optional fence that will be signaled when the
    // command buffers finshed execution. Due to this, we know when it is safe
    // for the command buffer to be reused. On the next frame CPU will wait for
    // this command to finish executing before it records new commands into it.
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    // Specify which semaphores to wait on before presentation can happen
    // We want to wait on command buffer to finish execution, then draw
    // Take signalSemaphore which will be signalled and wait on them.
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    // specify the swap chains to present images to and the index of image
    // for each swap chain. (Almost always a single one).
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    // this allows you to specify an array of VkResult values to check for
    // every individual swap chain if presentation was successful.
    // Not necessary if you're only usinf a single swap chain (just use the
    // return value of the present function).
    presentInfo.pResults = nullptr; // Optional

    // This submits the request to present an image to the swap chain.
    // Error handling for vkAcquireNextImageKHR and vkQueuePresentKHR
    // will be added later, their failure does not necessarily mean the program
    // should terminate.
    // It returns the same result as vkAcquireNextImageKHR with the same
    // meaning.
    result = vkQueuePresentKHR(presentQueue, &presentInfo);

    // also check for window resized
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
        framebufferResized) {
      // reset after vkQueuePresentKHR to ensure that the semaphores are in a
      // consistent state otherwise a signalled semaphore may never be properly
      // waited upon
      framebufferResized = false;
      // recreate swap chain if outdate or suboptimal for the best possible
      // result
      recreateSwapChain();
    } else if (result != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    // increment current frame, loop around (0->1->0)
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  // Generates a new tramsformation every frame to make the geometry spin around
  // Depends on GLM/matrix_transform and chrono
  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    // Time since rendering has started
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    // Define the model, view and projection transformations in the ubo.
    // Simple rotation about the Z-axis using the time variable
    UniformBufferObject ubo{};
    // model tranformations
    // glm::rotate takes an existing transformation, rotationangle and rotation
    // axis as params glm::mat4(1.0f) is an identity matrix, rotation angle is
    // 90 degrees per second.
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));

    // view transformations
    // look at the geometry from above at a 45 degree angle. params are: eye
    // position, center position and up axis parameters.
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));

    // projection transformations
    // Using a perspective projection with a 45 degree vertical FOV.
    // other params are: aspect ratio, near  and far view view planes. We used
    // the current swap chain extent to calculate aspect ratio, this takes new
    // size into account incase of a resize.
    ubo.proj = glm::perspective(
        glm::radians(45.0f),
        swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

    // flip Y coordinate
    // GLM was designed for OpenGL, where the Y coordinate of clip coordinates
    // is inverted.
    ubo.proj[1][1] *= -1;

    // copy data in ubo to the current uniform buffer.
    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
  }

  // Made static because GLFW does not know how to properly call a member
  // function with the right this pointer to our instance. But we stored a
  // pointer in GLFWwindow using glfwSetWindowUserPointer and can retrieve it to
  // access member variables.
  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    // retrieve stored `this` pointer from within the window
    auto app = reinterpret_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initWindow() {
    glfwInit();
    // Window hints
    glfwWindowHint(GLFW_CLIENT_API,
                   GLFW_NO_API); // originally designed for OpenGL context, but
                                 // we don't want this

    // Actual window
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr,
                              nullptr); // width, height, window title, monitor
                                        // to open window, OpenGL specific
    // store an arbitrary pointer in the window
    glfwSetWindowUserPointer(window, this);
    //  GLFW function to detect resize event, pass a callback  to it
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }
  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createTextureImage();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }

  void mainLoop() {
    // keeps the program running until there is either an error or the window is
    // closed
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }
    // Wait for the logical device to finish operations before exiting mainLoop
    // and destroying the window.
    vkDeviceWaitIdle(device);
  }

  void cleanup() {
    // End of program, no synchronization is necessary
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    // Delete after rendering, but before render pass it is based on
    cleanupSwapChain();

    // main texture image is used until end of the program
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);

    // uniform data will be used for draw calls, buffer should be destroyed when
    // we stop rendering
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroyBuffer(device, uniformBuffers[i], nullptr);
      vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    // will also clean up descriptor sets
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    // The descriptor layout should stick around while we may create new
    // graphics pipelines.
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);

    // vertex buffer should be available for use for rendering comands until the
    // end of the program. No dependance on swap chain.
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    // Free memory associate with buffer after buffer has been destroyed
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    // Destroy pipeline
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    // Destroy pipelineLayout
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    // Destroy render pass object after use throughout the program
    vkDestroyRenderPass(device, renderPass, nullptr);

    // Logical devices don't interact directly with instances. Destroyed alone
    vkDestroyDevice(device, nullptr);

    // Destroy Debug messenger if it exists
    if (enableValidationLayers) {
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }

    // Destroy surface before instance
    vkDestroySurfaceKHR(instance, surface, nullptr);
    // cleanup once the window is closed
    vkDestroyInstance(instance,
                      nullptr); // VkInstance should only be destroyed on
                                // program exit, deallocator is ignored here

    glfwDestroyWindow(window);

    glfwTerminate();
  }
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
