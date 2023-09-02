#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>  //may be needed for cstring

#include <vector>
#include <algorithm>
#include <string>
#include <string_view>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

//Provided standard diagnostics validation layer bundled in the Vulkan SDK
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;       
#else
    const bool enableValidationLayers = true;       //Runs in debug build
#endif


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

        /* Now redundant
        std::uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        //filling the Extension count info
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        */

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
        
        /* Now Redundant
        //checking for extension support, retrieving list of extensions
        std::uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);  //passing nullptr just requests the number of extensions

        //allocate array to hold extension details
        std::vector<VkExtensionProperties> extensions(extensionCount);

        //Query the extension details
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        //listing the extensions
        std::cout << "Available Vulkan extensions:\n";

        //VkExtensionProperties contains name and version of extension
        for (const auto& extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }

        //checking if glfwGetRequiredInstanceExtensions match the available extensions
        auto isPresent = [&extensions](std::string_view requiredExt) -> std::string {
            auto found =
            std::find_if(extensions.begin(), extensions.end(), [&requiredExt](const auto& extension){
                return strcmp(requiredExt.data(), extension.extensionName);
            });
            
            return (found != extensions.end())?  "Present" : "Missing";            
        };

        std::cout << "\nGLFW Required Extensions:\n";

        for (uint32_t i = 0; i < glfwExtensionCount;i++) {
            std::cout << '\t' << glfwExtensions[i] << " - " <<isPresent(glfwExtensions[i]) <<  std::endl;
        } */
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
    }

    void mainLoop() {
        //keeps the program running until there is either an error or the window is closed
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        //Destroy Debug messenger if it exists
        if(enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }


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
