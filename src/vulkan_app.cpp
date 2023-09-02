#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include <vector>
#include <algorithm>
#include <string>
#include <string_view>


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;


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

    //creating an instance involves specifing some details about the application to driver
    void createInstance() {
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

        //specifying the global extensions
        std::uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        //filling the Extension count info
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        //global validation layers to enable
        createInfo.enabledLayerCount = 0;
        //createInfo.ppEnabledLayerNames = ??   will be discussed later and filled

        //Everything is ready to create an instance, VkResult is result, check for failure
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

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
    }

    void mainLoop() {
        //keeps the program running until there is either an error or the window is closed
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
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
