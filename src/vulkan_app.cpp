#define GLFW_INCLUDE_VULKAN    //GLFW will include Vulkan and do some checks
#include <GLFW/glfw3.h>


#include <iostream>
#include <stdexcept>
#include <cstdlib>  
#include <cstdint>   // for std::uint32_t
#include <limits>   //Necessary for std::numeric_limits
#include <cstring>  //may be needed for cstring

#include <vector>
#include <algorithm>  //necessary for std::clamp
#include <string>
#include <string_view>
#include <map>
#include <optional>
#include <set>
#include <fstream>



const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

//Loading Shaders
//Reads all bytes from specified file and return a byte array.
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file (filename, std::ios::ate | std::ios::binary); //start reading at the end of file.

    if(!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    //determine size of file  using read position
    std::size_t fileSize = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(fileSize);  //allocate buffer to file size

    //seek to beginning 
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    //close file and return bytes
    file.close();
    return buffer;
}

//Provided standard diagnostics validation layer bundled in the Vulkan SDK
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

//list of required device extensions
//VK_KHR_SWAPCHAIN_EXTENSION_NAME is a macro defined as VK_KHR_swapchain
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;       
#else
    const bool enableValidationLayers = true;       //Runs in debug build
#endif


//Stores properties of window surface for swapchain creation (They must be compatible)
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


//stores different Message Queue Indices
struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;
    
    bool isComplete() {
        //families supporting drawing and presentation may not overlap, we want both to be supported
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

//Create vkDebugUtilsMessengerEXT object
VkResult CreateDebugUtilsMessengerEXT (VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                        const VkAllocationCallbacks* pAllocator,
                                        VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) { //instance is passed since debug messenger is specific it and its layers
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {  //return nullptr if it couldn't be loaded
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

}


//can be static 
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
                const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if( func != nullptr) {
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
    GLFWwindow* window;   //stores window for us.
    VkInstance instance;    //connection between your application and the Vulkan library
    VkDebugUtilsMessengerEXT debugMessenger;     //tell Vulkan about callback,
    VkSurfaceKHR surface;  //abstract type of surface to present rendered images
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;    //stores selected graphics card, implicitly destroyed along with VKInstance
    VkDevice device;   //stores logical device handle
    VkQueue graphicsQueue; //will store handle to the graphics queue, implicitly cleaned up on device destruction
    VkQueue presentQueue; //will store handle to present queue
    VkSwapchainKHR swapChain; //handler for swap chain object
    std::vector<VkImage> swapChainImages;   //Used to store handles to VkImages in the Swapchain
    
    VkFormat swapChainImageFormat;  //Store chosen format for swap chain images
    VkExtent2D swapChainExtent;     //Store chosen extent for swap chain images
    std::vector<VkImageView> swapChainImageViews;   //member to store the images views in.

    //creating an instance involves specifing some details about the application to driver
    void createInstance() {

        //check validation layers support first:
        if(enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }



        VkApplicationInfo appInfo{};  //optional but useful to help driver optimize application (e.g.if it uses well known engine)
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;   // you specify type of Vulkan struct here
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);  //use VK_MAKE_API_VERSION
        appInfo.apiVersion = VK_API_VERSION_1_0;
        //appInfo.pNext  = ??;   //It can point to extension information in the future, ignoring it, value initializes it to nullptr


        VkInstanceCreateInfo createInfo{};  //required, tells driver which global x10sion and validation layer to use
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        //Get required extensions, Specified Enabled extensions
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        //creation information for our debug messenger, placed outside to prevent destruction before instance is created
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};   

        //global validation layers to enable
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            //Added 
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;   //pass info to instance creation infor struct
        } else {
            createInfo.enabledLayerCount = 0;
        }

        //Everything is ready to create an instance, VkResult is result, check for failure
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
        

    }

    bool checkValidationLayerSupport() {
        std::uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);  //used just like vkEnumerateInstanceExtensionProperties above

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        //check if all the layer in validationLayers exist in availableLayers list
        for( const char* layerName : validationLayers) {
            bool layerFound = false;

            for(const auto& layerProperties : availableLayers) {
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

    //Return the required list of extension based on whether validation layers are enabled or not,
    std::vector <const char*> getRequiredExtensions() {
        std::uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);  //std::vector<T>(InputIt first, InputIt last)

        //conditionally add the debug messenger extension
        //if you have this Error: VK_ERROR_EXTENSION_NOT_PRESENT, some validation layer is missing
        if(enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME); //equal to VK_EXT_debug_utils string literal , this avoids typos
        }

        return extensions;
    }


    //Debug callback function
    //VKAPI_ATTR and VKAPI_CALL ensure function has the right signature for Vulkan to call it
    //has the PFN_vkDebugUtilsMessengerCallbackEXT prototype
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback (
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
            return VK_FALSE;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;       //All except message severity verbose
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;   //enabled all here
        createInfo.pfnUserCallback = debugCallback;     //pointer to user defined callback function
        createInfo.pUserData = nullptr; //Optional
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        //fill structure with details about the messenger and its callback
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);
        

        //Call function to function to create extensionobject if available
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    //helper function for pickPhysicalDevice()
    //Checks if passed device is suitable for the operation we want to perform
    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        //Check for swapChain Adequacy
        bool swapChainAdequate = false;
        if (extensionsSupported) {  //only query for swap chain support after verifying the extension is available
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            //for now we want at least one supported image format and one supported presentation mode for the given surface
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();    
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }


    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        std::uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);  //fills the extensionCount

        //All vector to store all available extensions
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        //Copy required extensions defined globally
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);  //erase will succeed if present
        }

        return requiredExtensions.empty();  //if empty, extension is present
    }
    //Select a graphics card that supports the features we need. Select only one for now
    void pickPhysicalDevice() {
        //Listing the graphics cards is very similar to listing extensions and starts with querying just the number.
        std::uint32_t deviceCount{0};
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);  //stores the number of extension here

        if(deviceCount == 0) { //0 devices with Vulkan support no need for going further.
            throw std::runtime_error("failed to find GPUs with Vulkan support");
        }

        //allocate array to hold alll VkPhysicalDevice handles
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        //Check for Device suitability
        for (const auto& device : devices) {
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
        //Logic to find graphics queue family
        QueueFamilyIndices indices;
        
        //Extract Queue Families Properties, similar pattern to  Physical Device and Extensions
        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device,  &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());


        // Assign index to queue families that could be found

        int i{0};
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) { //support drawing commands
                indices.graphicsFamily = i;
            }

            //Look for queue family that can present to our window surface.
            VkBool32 presentSupport = false;
            //takes in physical device, queue family index and surface
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport); //populates presentSupport if yes.
            if (presentSupport) {
                indices.presentFamily = i;
            }

            //break early
            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    void createLogicalDevice() {
        //Get Queue Family index
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);  //pass in handle to the current physical device

        //CreateInfos for creating queues from the Queue families
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        //using a set of all unique queue families necessary for required queue
        //value certain due to check in pickPhysicalDevice()
        std::set<std::uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};  

        //populate Device Queue Create info
        float queuePriority = 1.0f;
        for (std::uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;

            //Assign priority for scheduling command buffer execution - Required
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        //Specify used device features
        VkPhysicalDeviceFeatures deviceFeatures{}; //we don't need anything special, so just define for now.

        //Create the logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        //Add pointers to the queue creation info and device features structs
        createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());;
        createInfo.pQueueCreateInfos = queueCreateInfos.data(); //points to all createInfos
        createInfo.pEnabledFeatures = &deviceFeatures;

        //The rest of info is like VkInstanceCreateInfo but device specific
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {  
            //ignored by current Vulkan implementation (Uses Instance specific Vlaidation). 
            //Include for compatiblity reasons
            createInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        //No need for device specific extensions for now


        //Instantiate logical device
        //physical device to inteface with, queue and usage info, optional allocator callback,
        //pointer to variable to store the logical device handle in.
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        //Retrieves queue handles for each queue family.
        //Passed logical device, queue family, queue index (0 in this case, since we are only creating one),
        //pointer to variable to store queue handle in
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);

        //If the queue families are the same, then we only need to pass its index once(both handles will have the same value) 
        //call to retrieve the queue handle:
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    
    //Used to create the VkSurface_KHR instance
    void createSurface() {
        //Takes simple paratmeters instead of a struct
        //VkInstance, GLFW window pointer, custom allocator and pointer to VksurfaceKHR
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    //Settings for Swap Chain
    
    //Populates SwapChainSupportDetails struct
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        //Basic surface capabilities: returned into a VkSurfaceCapabilitiesKHR struct
        // Takes in specified VkPhysicalDevice adn VkSurfaceKHR window surface intom account.
        //These two will be in all support querying functions as they are core components of the swap chain.
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        //Query for surported surfacr formats. List of structs. So 2 function calls
        std::uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if(formatCount != 0) {
            details.formats.resize(formatCount); //make sure the vector is resized to all available formats.
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        //Query the presentation modes (Same procedure)
        std::uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if(presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }

    //Surface format
    //Pass the formats member of SwapChainSupportDetails
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        //Each entry in vector contains format and a colourSpace member.
        //format specifies the color channels and types, e.g 
        //VK_FORMAT_B8G8R8A8_SRGB stores B G R A channels i 8Bit (32Bits per pixel)
        //colorSpace indicates if SRGB colour space is supported using the VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag.
        //or VK_COLORSPACE_SRGB_NONLINEAR_KHR in old versions of the spec

        //Go through list and see if preferred combination is available
        //VK_FORMAT_B8G8R8A8_SRGB because SRGB results in better conceived colours and is a common standard
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == 
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        //Just choose the first if the most preferred is not availble
        return availableFormats[0];
    }

    //Presentation Mode
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {

        for (const auto& availablePresentMode : availablePresentModes) {
            //Nice trade-off if energy usage is mot a concern. Minimizes tearing while maintaining low latency
            //Up to date New images are rendered until the vertical blank
            //Doesn't block application when queue is full, replace old images in queue with newer ones. 
            if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR ) { //aka Tripple Buffering
                return availablePresentMode;
            }
        }

        //Guaranteed to be available
        //Swap chain is a queue, FIFO is observed. Similar to Vertical sync. After refresh is called Vertal blank
        return VK_PRESENT_MODE_FIFO_KHR;
    }


    //Swap Extent
    //VkSurfaceCapabilitiesKHR contains the range of possible resolution of window in pixels
    //We can't use WIDTH and HEIGHT declared because it can vary. Also, that is in screen coordinates unit (not pixels)
    VkExtent2D chooseSwapExtent (const VkSurfaceCapabilitiesKHR& capabilities) {
        //Use currentExtent member of VkSurfaceCapabilitiesKHR, unless 
        //Some window managers set the currentExtent to maximum value of uint32_t as a special case(else block)
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        } else { //Pick resolution that best matches window within minImageExtent and maxImageExtent in pixels
            int width{0}, height{0};
            //Returns the current height of the window( window's frame buffer) in pixels
            glfwGetFramebufferSize(window, &width, &height); 

            //Create actualExtent struct with the size from glfw window in pixels
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width), 
                static_cast<uint32_t>(height)
            };

            //Clamps value to used to bound the values between the allowed min and max supported by the implementation
            //returns passed value unless it is higher or lower than the bounds, then it returns the closest bound.
            actualExtent.width   = std::clamp(actualExtent.width, capabilities.minImageExtent.width,
            capabilities.minImageExtent.width);
            actualExtent.height  = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
            capabilities.minImageExtent.height);

            return actualExtent;
        }
    }

    void createSwapChain() {
        //Get Swapchain suport info
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice); //pass in physical device handler

        //Choose surface formats
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);

        //Choose presentation mode
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);

        //Get Screen Extents (in pixels)
        VkExtent2D extent  = chooseSwapExtent(swapChainSupport.capabilities);


        //Decide how many images we would like to have in the swap chain
        //Impleemntation specifies the minimum number it requires to fucntion
        //To avoid waiting on driver to complete internal operations before acquiring another images to render
        //request at least one more image than minimum
        std::uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        //Don't exceed maximum number of images
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }


        //fill in the creation structure
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;  // pass in the surface handler

        //specify details of the swap chain images
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;

        //imageArrayLayers specifies the amount of layers each images consists of
        //Always 1 unless you are developing steroscopic 3D application
        createInfo.imageArrayLayers = 1; 

        //imageUsage Specifies what kind of operations we use the images in the swap chain for.
        //For now we render directly to them i.e.  they are used as color attachment
        //You can render to a separate image first to perform operation like post processing
        //In that case use VK_IMAGE_USAGE_TRANSFER_DST_BIT and 
        //then use a memory op to transfer the rendered image to a swap chain image
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        //Specify how to handle swap chain images used across multiple queue families.
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily) {
        // graphics queue family is different from the presentation queue
        // Drawing on images in the swap chain from the graphics queue and submit to presentation queue
            //Images can be used across multiple queue families without explicit owership transfers
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT; //doing this here for ease, later we can ownership control
    
            //Concurrent mode requires you specify in advancem between with queue families ownership will be shared
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            //image owned by one queue family at a time,must be transfered before using it in another family.
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;  //Offers best performance
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        //Specify that a certain transform should be applied to images in the swap chain 
        //if it is supported by setting to capabilities.supportedTransforms, otherwise specify current transformation
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform; 

        //Specifies if the Alpha channel should be used for blending with other windows in the window system. We ignore it here.
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        createInfo.presentMode = presentMode;  //pass the presentation mode handler
        createInfo.clipped = VK_TRUE; //TRUE means we don't care about the color of obscured pixels. Use FALSE unless it is really needed

        //Assume we only ever create one swap chain
        createInfo.oldSwapchain = VK_NULL_HANDLE;


        //Create the swapChain
        //Pass in device, creation info, custom allocator (nullptr) for now, and the handler to store it in
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        //Retrieve the swap chain images
        //use the count, resize pattern 
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data()); //store the images


        //Store chosen Format and Extent
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size()); //resize to fit all of the image views we'll be creating

        //Iterate over all of the swap chain images
        for(std::size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];

            //Specify how the image date should be interpreted.
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;

            //The component field allow you to swizzle the color channels around.
            //e.g map all the channels to red channel for a monochrome texture or
            //map constant values of 0 and 1 to a channel
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY; //stick to default mapping
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            //Define the image's purpose and part of image should be accessed
            //Use images as color targets and no mipmapping levels or multiple layers
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;  //used a color target
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;


            //Call creation function (each should be explicitly destroyed)
            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    //Creates graphics pipeline, loads shaders code
    void createGraphicsPipeline() {
        //load bytecode of the two shaders:
        auto vertShaderCode = readFile("../../shaders/vert.spv"); //path relative to the executable
        auto fragShaderCode = readFile("../../shaders/frag.spv");

        std::cout << "Check for correct load\n";
        std::cout << "Loaded vertex shader bytes: " << vertShaderCode.size() << "\n";
        std::cout << "Loaded shader shader bytes: " << fragShaderCode.size() << "\n";

        //Load ShaderModules:
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        //Programmable stage
        //Create Pipeline Shader stage for Vertex Shader
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        //tells vulkan in which pipeline stage shader will be used. 
        //Enums for other programmable stages exist
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; 
        //specify the shader module containing the code
        vertShaderStageInfo.module = vertShaderModule;
        //The shader code entry point
        vertShaderStageInfo.pName = "main";

        //Create Pipeline Shader stage for Fragment Shader
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";


        //Array to contain two structs
        //Would be referenced in the pipeline creation
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};


        //Fixed function stage

        //Shader Modules clean up
        vkDestroyShaderModule (device, fragShaderModule, nullptr);
        vkDestroyShaderModule (device, vertShaderModule, nullptr);
    }

    //Create VkShaderModule to wrap code passed into the graphics pipeline
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        //specify pointet to the buffer witht he bytecode and length
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        //pCode is a uint32_t pointer
        //reinterpret cast, data alignment assured for std::vector
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data()); 

        //create ShaderModule
        VkShaderModule shaderModule; //only needed for pipeline creation
        if(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return  shaderModule;
    }
    
    void initWindow() {
        glfwInit();
        //Window hints
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);    //originally designed for OpenGL context, but we don't want this 
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);      //Too complicated, will handle later.

        //Actual window
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);  //width, height, window title, monitor to open window, OpenGL specific
    }
    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createGraphicsPipeline();
    }

    void mainLoop() {
        //keeps the program running until there is either an error or the window is closed
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        //Destroy views
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        //Destroy swapchain before the device
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        //Logical devices don't interact directly with instances. Destroyed alone
        vkDestroyDevice(device, nullptr);

        //Destroy Debug messenger if it exists
        if(enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        //Destroy surface before instance
        vkDestroySurfaceKHR(instance, surface, nullptr);
        //cleanup once the window is closed
        vkDestroyInstance(instance, nullptr);   //VkInstance should only be destroyed on program exit, deallocator is ignored here

        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

