#define GLFW_INCLUDE_VULKAN // GLFW will include Vulkan and do some checks
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // use Vulkan 0.0 to 1.0 not OpenGL's -1
                                    // to 1.0
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES // To help with alignments
                                           // requirements
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // exposes functions that can be used to generate MVP transformations.

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
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Vertex data

// To be used in the vertex shader
// GLM provides C++ types that exactly match vector types used
// in shader language
struct Particle {
  glm::vec2 position;
  glm::vec2 velocity;
  glm::vec4 color;

  // A vertex binding describes at which rate to load data from memory
  // throughout the vertices. It specifies the number of bytes between data
  // entries and whether to move to the next data entry after each vertex or
  // after each instance.
  static VkVertexInputBindingDescription getBindingDescription() {
    // All of our per-vertex data is packed together in one array, so we're
    // only going to have one binding.
    VkVertexInputBindingDescription bindingDescription{};

    // This specifies the index of the binding in the array of bindings
    bindingDescription.binding = 0;
    // This specifies the number of bytes from one entry to the next
    bindingDescription.stride = sizeof(Particle);
    // This determines how you mobe to the next data, either move after each
    // vertex or after each instance. We are moving after each instance.
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  // An attribute description struct describes how to extract a vertex attribute
  // from a chunk of vertex data originating from a binding description. We have
  // two attributes, position and color. Note that we don't add velocity to the
  // vertex input attributes, as this is only used by the compute shader.
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
    attributeDescriptions[0].offset = offsetof(Particle, position);

    // color attribute is described similarly
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Particle, color);

    return attributeDescriptions;
  }
};

// Uniform buffer object  descriptor
// GLM data types exactly matches the defintion in the shader
// This is good for bianry compatibilty and operations like
// memcpy a UBO to a VkBuffer.
struct UniformBufferObject {
  float deltaTime = 1.0f;
};

// Windowing and Vulkan attributes
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const uint32_t PARTICLE_COUNT = 8192;

// Number of frames to be processed concurrently
const int MAX_FRAMES_IN_FLIGHT = 2;

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
  std::optional<std::uint32_t> graphicsAndComputeFamily;
  std::optional<std::uint32_t> presentFamily;

  bool isComplete() {
    // families supporting drawing, compute and presentation may not overlap, we
    // want both to be supported
    return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
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

class ComputeShaderApplication {
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
      VK_NULL_HANDLE; // stores selected graphics card, implicitly destroyed
                      // along with VKInstance
  VkDevice device;    // stores logical device handle

  VkQueue graphicsQueue; // will store handle to the graphics queue, implicitly
                         // cleaned up on device destruction
  VkQueue computeQueue;  // will store handle to the compute queue
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
  VkPipelineLayout pipelineLayout;
  VkDescriptorSetLayout
      computeDescriptorSetLayout; // contains all descriptor bindings.
  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet>
      computeDescriptorSets; // holds the descriptor set handles
  VkPipelineLayout computePipelineLayout;
  VkPipeline graphicsPipeline; // holds the Graphics Pipeline object
  VkPipeline computePipeline;

  VkCommandPool commandPool; // Command pools manage memory used to store the
                             // buffer. Command buffers are allocated from them.
  std::vector<VkCommandBuffer> commandBuffers; // To store CommandBuffers
  std::vector<VkCommandBuffer> computeCommandBuffers;

  std::vector<VkSemaphore>
      imageAvailableSemaphores; // semaphore signals image has been acquired
                                // from swapchain, ready for rendering.
  std::vector<VkSemaphore>
      renderFinishedSemaphores; // semaphore signals rendering finished and it
                                // ready for presentation.
  std::vector<VkSemaphore> computeFinishedSemaphores; // compute semaphores

  std::vector<VkFence> inFlightFences; // fence to make sure only one frame is
                                       // rendering at a time.
  std::vector<VkFence> computeInFlightFences;

  std::vector<VkBuffer>
      uniformBuffers; // holds the multiple uniform buffer for frames in flight
  std::vector<VkDeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  std::vector<VkBuffer>
      shaderStorageBuffers; // holds the multiple shader storage buffer for
                            // frames in flight
  std::vector<VkDeviceMemory> shaderStorageBuffersMemory;

  std::uint32_t currentFrame = 0; // frame index to keep track of current frame.

  bool framebufferResized = false; // true when a resize happens

  float lastFrameTime = 0.0f; // keeps track of last frame time.

  double lastTime = 0.0f;

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
    appInfo.pApplicationName = "Compute Shader Example";
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
  // Loading Shaders
  // Reads all bytes from specified file and return a byte array.
  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(
        filename,
        std::ios::ate | std::ios::binary); // start reading at the end of file.

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

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate &&
           supportedFeatures.samplerAnisotropy;
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

    // Check for Device suitability and set max samples for suitable device
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
      if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          (queueFamily.queueFlags &
           VK_QUEUE_COMPUTE_BIT)) { // support both drawing & compute commands
        indices.graphicsAndComputeFamily = i;
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
        indices.graphicsAndComputeFamily.value(),
        indices.presentFamily.value()};

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
    VkPhysicalDeviceFeatures deviceFeatures{};

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
    vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0,
                     &graphicsQueue);

    // Get compute queue incase it is different from graphics queue
    vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0,
                     &computeQueue);

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
    std::uint32_t queueFamilyIndices[] = {
        indices.graphicsAndComputeFamily.value(),
        indices.presentFamily.value()};

    if (indices.graphicsAndComputeFamily != indices.presentFamily) {
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

  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags,
                              std::uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;

    // Specify how the image date should be interpreted.
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;

    // The component field allow you to swizzle the color channels around.
    // e.g map all the channels to red channel for a monochrome texture or
    // map constant values of 0 and 1 to a channel
    // Note this can be left out as VK_COMPONENT_SWIZZLE_IDENTITY is defined as
    // 0.
    viewInfo.components.r =
        VK_COMPONENT_SWIZZLE_IDENTITY; // stick to default mapping
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    // Define the image's purpose and part of image should be accessed
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    // single layers
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    // Call creation function (each should be explicitly destroyed)
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create image views!");
    }
    return imageView;
  }

  void createImageViews() {
    swapChainImageViews.resize(
        swapChainImages
            .size()); // resize to fit all of the image views we'll be creating

    // Iterate over all of the swap chain images
    for (std::size_t i = 0; i < swapChainImages.size(); i++) {
      // Use images as color targets
      swapChainImageViews[i] =
          createImageView(swapChainImages[i], swapChainImageFormat,
                          VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
  }

  void createRenderPass() {
    // color buffer attachment represented by one of the images of the swap
    // chain
    VkAttachmentDescription colorAttachment{};
    // format of color attachment should match the format of the
    // swap chain images.
    colorAttachment.format = swapChainImageFormat;
    // For multisampling
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
    // If you want the image to be ready for presentation using the swapchain
    // after rendering we use VK_IMAGE_LAYOUT_PRESENT_SRC_KHR i.e  Images to be
    // present in the swap chain.
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
    // Added a reference to the attachment for the first (and only) subpass'
    // unlike color attachments, a subpass can only use a single depth(+stencil)
    // attachment. Doing depth tests on multiple buffers doesn't make sense
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
    // Wait on swap chain to finish reading from image before we can access it.
    // so wait on the color attachment output stage itsel.
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;

    // The operations that should wait on this are in the color attachment
    // stage. These prevent the transition from happening
    // until it's actually necessary or allowed: when we want to start writing
    // colors to it.
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Allow write operations, since we have a load operation that clears
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

  void createComputeDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};
    layoutBindings[0].binding = 0;
    // Number of values in the array
    // Can be used the specify a transformation for each bones in a skeleton for
    // skeletal animation for example. Our MVP transformation is in a single
    // uniform buffer object so 1 is specified.
    layoutBindings[0].descriptorCount = 1;
    // Uniform buffer object
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // only relevant for image sample related descriptors.
    layoutBindings[0].pImmutableSamplers = nullptr;
    // the shader stage it is going to be referenced. It can be a combination of
    // VkShaderStageFlagBits values or VK_SHADER_STAGE_ALL_GRAPHICS
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Two bindings because we want to use the delta time between them for the
    // particle calculations.  Particle positions are updated frame by frame
    // based on a delta time. Each frame needs to know about the
    // last frames' particle positions
    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[1].pImmutableSamplers = nullptr;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorCount = 1;
    layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[2].pImmutableSamplers = nullptr;
    layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Create VkDescriptorSetLayout
    // VkDescriptorSetLayoutCreateInfo
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                    &computeDescriptorSetLayout) !=
        VK_SUCCESS) {
      throw std::runtime_error(
          "failed to create compute descriptor set layout!");
    }
  }

  // Creates graphics pipeline, loads shaders code
  void createGraphicsPipeline() {
    // load bytecode of the two shaders:
    auto vertShaderCode =
        readFile("resources/shaders/vert.spv"); // path relative to the
                                                // executable
    auto fragShaderCode = readFile("resources/shaders/frag.spv");

#ifndef NDEBUG
    std::cout << "Check for correct load\n";
    std::cout << "Loaded vertex shader bytes: " << vertShaderCode.size()
              << "\n";
    std::cout << "Loaded shader shader bytes: " << fragShaderCode.size()
              << "\n";
#endif
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
    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

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
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
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
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    // Enable sample shading in the pipeline to improve image quality
    // There is some performance cost
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // color blending
    // configuration per attached framebuffer
    // Allows configuring this method of color blending: Mix the old and new
    // value to produce a final color
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

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
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
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

  void createComputePipeline() {
    auto computeShaderCode =
        readFile("resources/shaders/compute.spv"); // relative to the executable
#ifndef NDEBUG
    std::cout << "Check for correct load\n";
    std::cout << "Loaded Compute shader bytes: " << computeShaderCode.size()
              << "\n";
#endif
    // Load ShaderModules:
    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    // Programmable stage
    // Only one shader stage
    // Create Pipeline Shader stage for Compute Shader
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    // specify the shader module containing the code
    computeShaderStageInfo.module = computeShaderModule;
    // The shader code entry point
    computeShaderStageInfo.pName = "main";

    // Pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // Tell Vulkan which descriptors the shaders will be using
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
    // Push constants are another way of passing dynamic values to shaders
    pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    // Create pipelineLayout object
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &computePipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create pipeline layout!");
    }

    // Creating the Compute Pipeline
    //  Reference VkPipelineShaderStageCreateInfo structs
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = computePipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    // Create Compute Pipeline
    // Note that the function is designed to take multiple
    // VkComputePipelineCreateInfo objects and create multiple VkPipeline
    // object in a single call The second parameter was VK_NULL_HANDLE argument,
    // references an optional VkPipelineCache object to help reuse data across
    // mutliple calls of the function
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                 nullptr, &computePipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create compute pipeline!");
    }

    // Shader Module clean up
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
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
      // bind the dpeth image to the depth attachment as a second attachment
      // The color attachment differs for every swap chain image, but the same
      // depth image can be used by all of them because only a single subpass is
      // running at the same time due to semaphores
      std::array<VkImageView, 1> attachments = {swapChainImageViews[i]};

      // Create Framebuffer
      VkFramebufferCreateInfo framebufferInfo{};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      // specify with which renderPass, the framebuffer needs to be compatible
      // with. render pass must be compatible with framebuffer usually by having
      // the same number and type of attachments.
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount =
          static_cast<std::uint32_t>(attachments.size());
      framebufferInfo.pAttachments = attachments.data();
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
    poolInfo.queueFamilyIndex =
        queueFamilyIndices.graphicsAndComputeFamily.value();

    // Create CommandPool
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  // Takes a list of candidate formats in order from most desirable to least
  // desirable and checks which is the first one that is supported
  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
    // support of a format depends on the tiling mode and usage
    for (VkFormat format : candidates) {
      // contains 3 fields, only linearTilingFeatures and optimakTilingFeatures
      // are relevant here
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

      // The one we check depends ont he tiling parameter of the function
      if (tiling == VK_IMAGE_TILING_LINEAR &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }
    // We can either return a special value or throw an exception
    throw std::runtime_error("failed to find supported format!");
  }

  // Select a format with a depth component that supports usage as depth
  // attachment
  VkFormat findDepthFormat() {
    // we make sure to use VK_FORMAT_FEATURE FLAG rather than VK_IMAGE_USAGE.
    return findSupportedFormat({VK_FORMAT_D32_SFLOAT,
                                VK_FORMAT_D32_SFLOAT_S8_UINT,
                                VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  // chosen format contains stencil attachment?
  bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT;
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
      throw std::runtime_error("failed to create buffer!");
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
      throw std::runtime_error("failed to allocate buffer memory!");
    }

    // Once successful, associate this memory with the buffer
    // The fourth param is the offset within the region of memory.
    // If offset is non-zero, it is required to be divisible by
    // memRequirements.alignment
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
  }

  // helper function for copyBuffer
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

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    // This function transfers the contents of the buffer.
    // It also takes the array of regions to copy.
    // The regions are defined in VkBufferCopy struct. It is not possible to
    // specify VK_WHOLE_SIZE here.
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
  }

  void createShaderStorageBuffers() {
    // Initialize particles and move the initial particle info to GPU
    // host side
    std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
    std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

    // Initial particle positions on a circle
    std::vector<Particle> particles(PARTICLE_COUNT);
    for (auto &particle : particles) {
      float r = 0.25f * std::sqrt(rndDist(rndEngine));
      float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
      float x = r * std::cos(theta) * HEIGHT / WIDTH;
      float y = r * std::sin(theta);
      particle.position = glm::vec2(x, y);
      particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
      particle.color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine),
                                 rndDist(rndEngine), 1.0f);
    }

    VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

    // staging buffer in host memory to hold initial particle properties,
    // It is used to upload data to the gpu
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
    memcpy(data, particles.data(),
           static_cast<size_t>(bufferSize)); // Now memcpy
    vkUnmapMemory(device, stagingBufferMemory);

    shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

    // Using staging buffer as a source we then create the per-frame shader
    // storage buffers and copy the particle properties from the staging buffers
    // to each of these
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // usage is as a destination buffer for a transfer, a vertex buffer and
      // a storage buffer. memory is device local so can't use vkMapMemory but
      // can copy to it from a staging buffer.
      createBuffer(bufferSize,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shaderStorageBuffers[i],
                   shaderStorageBuffersMemory[i]);
      // Copy data from the staging buffer (host) to the shader storage buffer
      // (GPU)
      copyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
    }

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

  void createComputeCommandBuffers() {
    computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount =
        static_cast<std::uint32_t>(computeCommandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocInfo,
                                 computeCommandBuffers.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate compute command buffers!");
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
    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}}; // set color field
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

      VkDeviceSize offsets[] = {0};
      // Binds vertex buffers to bindings
      // second param is the offset, third param is the number of bindings we're
      // going to specify vertex buffer for.
      // last two are the array of vertex buffers to bind and byte offsets to
      // start reading from.
      vkCmdBindVertexBuffers(commandBuffer, 0, 1,
                             &shaderStorageBuffers[currentFrame], offsets);

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
      vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);
    // clang-format on

    // end render pass
    vkCmdEndRenderPass(commandBuffer);

    // Finish recording the command buffer
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }

  void recordComputeCommandBuffer(VkCommandBuffer commandBuffer) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error(
          "failed to begine recording compute command buffer");
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            computePipelineLayout, 0, 1,
                            &computeDescriptorSets[currentFrame], 0, nullptr);

    // Will dispatch the specified number of local work groups in the the x,yz
    // dimension respectively. Here it is only one dimensional in the x
    // direction. We divide the number of particles (in array) by 256 because in
    // the compute shader code,  we defined that every compute shader in a work
    // group will do 256 invocations. So for 4096 particles - 16 work groups,
    // each running 256 compute shader invocations. These numbers should be
    // profiled on.

    // If your particle size would be dynamic and can't always be divided by
    // e.g. 256, you can always use gl_GlobalInvocationID at the start of your
    // compute shader and return from it if the global invocation index is
    // greater than the number of your particles.
    vkCmdDispatch(commandBuffer, PARTICLE_COUNT / 256, 1, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to record compute command buffer");
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
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // We allocate one of these descriptors for every frame.
    poolSizes[0].descriptorCount =
        static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT);
    // Combined image sampler
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // double the number of VK_DESCRIPTOR_TYPE_STORAGE_BUFFER types  requested
    // from the pool because our sets reference the SSBOs of the last and
    // current frame.
    poolSizes[1].descriptorCount =
        static_cast<std::uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
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

  void createComputeDescriptorSets() {
    // specify the descriptor pool to allocate from
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               computeDescriptorSetLayout);
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

    computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    // Allocates descriptor sets, each one with one uniform buffer descriptor
    if (vkAllocateDescriptorSets(device, &allocInfo,
                                 computeDescriptorSets.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate descriptor sets!");
    }

    // Configure the descriptors within
    // Descriptors that refer to buffers, like our uniform buffer descriptor,
    // are configured with a VkDescriptorBufferInfo struct. This structure
    // specifies the buffer and the region within it that contains the data for
    // the descriptor.
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      VkDescriptorBufferInfo uniformBufferInfo{};
      uniformBufferInfo.buffer = uniformBuffers[i];
      uniformBufferInfo.offset = 0;
      // overwriting the whole buffer, you can also use the VK_WHOLE_SIZE value
      // to overwrite the whole buffer
      uniformBufferInfo.range = sizeof(UniformBufferObject);

      std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      // the descriptor set and binding to update
      descriptorWrites[0].dstSet = computeDescriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      // descriptor can be arrays but not used here.
      descriptorWrites[0].dstArrayElement = 0;
      // It's possible to update multiple descriptors at once in an array,
      // starting at index dstArrayElement
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      // how many array elements you want to update
      descriptorWrites[0].descriptorCount = 1;

      // These reference an array with descriptorCount structs that actually
      // configure the descriptor depending on the type you actually need to use
      // pBufferInfo is used for descriptors that refer to buffer data.
      // Our descriptor is based on buffers
      descriptorWrites[0].pBufferInfo = &uniformBufferInfo;
      descriptorWrites[0].pImageInfo = nullptr; // Optional
      //  pTexelBufferView is used for descriptors that refer to buffer views
      descriptorWrites[0].pTexelBufferView = nullptr; // Optional

      VkDescriptorBufferInfo storageBufferInfoLastFrame{};
      storageBufferInfoLastFrame.buffer =
          shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT];
      storageBufferInfoLastFrame.offset = 0;
      storageBufferInfoLastFrame.range = sizeof(Particle) * PARTICLE_COUNT;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = computeDescriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[1].descriptorCount = 1;
      //  pImageInfo is used for descriptors that refer to image data
      descriptorWrites[1].pImageInfo = nullptr;
      descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

      VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
      storageBufferInfoCurrentFrame.buffer = shaderStorageBuffers[i];
      storageBufferInfoCurrentFrame.offset = 0;
      storageBufferInfoCurrentFrame.range = sizeof(Particle) * PARTICLE_COUNT;

      descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[2].dstSet = computeDescriptorSets[i];
      descriptorWrites[2].dstBinding = 2;
      descriptorWrites[2].dstArrayElement = 0;
      descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptorWrites[2].descriptorCount = 1;
      //  pImageInfo is used for descriptors that refer to image data
      descriptorWrites[2].pImageInfo = nullptr;
      descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;

      // applies updates,  It accepts two kinds of arrays as parameters: an
      // array of VkWriteDescriptorSet and an array of VkCopyDescriptorSet. The
      // latter can be used to copy descriptors to each other, as its name
      // implies.
      vkUpdateDescriptorSets(
          device, static_cast<std::uint32_t>(descriptorWrites.size()),
          descriptorWrites.data(), 0, nullptr);
    }
  }

  // Creates semaphores and fences used in the program
  void createSyncObjects() {
    // resize sync objects
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

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
      if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &computeFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence(device, &fenceInfo, nullptr,
                        &computeInFlightFences[i]) != VK_SUCCESS) {
        throw std::runtime_error(
            "failed to create compute synchronization objects - "
            "semaphores and fences - for a frame!");
      }
    }
  }

  // Cleans up previous versions of swapchain, image views, depth resources and
  // framebuffers
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
    // Multisampling needed after resize or minimize
    createFramebuffers();
  }

  void drawFrame() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // Compute submission
    vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE,
                    UINT64_MAX);

    updateUniformBuffer(currentFrame);
    vkResetFences(device, 1, &computeInFlightFences[currentFrame]);

    vkResetCommandBuffer(computeCommandBuffers[currentFrame],
                         /*VkCommandBufferResetFlagBits*/ 0);
    recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

    // This submission to the compute queue updates the particle positions using
    // the compute shader
    if (vkQueueSubmit(computeQueue, 1, &submitInfo,
                      computeInFlightFences[currentFrame]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit compute command buffer!");
    }

    // Graphics submission
    // This submission will use that updated particle position data to draw the
    // particle system.

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

    // which semaphore to wait on before execution begins
    // Wait for the compute work to finish so it doesn't start fetching vertices
    // while the compute buffer is still updating them.
    VkSemaphore waitSemaphores[] = {computeFinishedSemaphores[currentFrame],
                                    imageAvailableSemaphores[currentFrame]};

    // which stage of the pipeline to wait
    // have the graphics submission wait on the
    // VK_PIPELINE_STAGE_VERTEX_INPUT_BIT stage, where vertices are consumed.
    // wait with writing colors to the image until it is available so specify
    // stage of graphics pipeline that writes to the color attachment.
    // Theoretically, implementation can already start executing our vertex
    // shader and such when image is not yet available.
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    // Submitting the command buffer
    // Queue submission and synchronization
    submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 2;
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

  void updateUniformBuffer(uint32_t currentImage) {
    UniformBufferObject ubo{};
    ubo.deltaTime = lastFrameTime * 2.0f;

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
    auto app = reinterpret_cast<ComputeShaderApplication *>(
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

    lastTime = glfwGetTime(); // used to calulate interframe time
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
    createComputeDescriptorSetLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createFramebuffers();
    createCommandPool();
    createShaderStorageBuffers();
    createUniformBuffers();
    createDescriptorPool();
    createComputeDescriptorSets();
    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();
  }

  void mainLoop() {
    // keeps the program running until there is either an error or the window is
    // closed
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
      // We want to animate the particle system using the last frames time to
      // get smooth, frame-rate independent animation
      double currentTime = glfwGetTime();
      lastFrameTime = static_cast<float>((currentTime - lastTime) * 1000.0);
      lastTime = currentTime;
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
      vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
      vkDestroyFence(device, computeInFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    // Delete after rendering, but before render pass it is based on
    cleanupSwapChain();

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
    vkDestroyDescriptorSetLayout(device, computeDescriptorSetLayout, nullptr);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vkDestroyBuffer(device, shaderStorageBuffers[i], nullptr);
      vkFreeMemory(device, shaderStorageBuffersMemory[i], nullptr);
    }

    // Destroy pipeline
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    // Destroy pipelineLayout
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);

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

int main(int argc, char *argv[]) {
  ComputeShaderApplication app;
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
