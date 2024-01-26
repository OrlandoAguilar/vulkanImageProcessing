/*
Copyright 2024 Orlando Aguilar Vivanco

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
This test is an example of using vulkan for image processing. It doesn't
render anything to the screen, instead it uses several compute shader to
perform different effects to an image, saving the output of the shader into
a png file
*/

#define VMA_IMPLEMENTATION

#include <Windows.h>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <filesystem>
#include <limits>
#include <fstream>
#include <iterator>
#include <vector>
#include <numeric>
#include <map>
#include <vma/vk_mem_alloc.h>

// stb image is used to read and write images
#define __STDC_LIB_EXT1__
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

//This project includes the capability of capturing a render doc capture from code for a single dispatch. 
//Useful for debugging compute only workload.
#ifdef CAPTURE_RENDERDOC
#include "renderdoc_app.h"
#endif


// Enum to reference the descriptor sets of the project. Only 3 descriptor sets are created and they are stored
// in a vector, the enum is used to acces the elements in the vector.
enum {
    INPUT_OUTPUT = 0,
    INPUT_INTERMEDIATE = 1,
    INTERMEDIATE_OUTPUT = 2
};

// structure containing the information of the images that are being passed to the shader. This data is passed to the shader
// as a push constant
struct imageData_t {
    uint32_t width;
    uint32_t height;
};


class Image {
private:
    int width;
    int height;
    const int neededChannels = 4;
    //unsigned char* image;
    std::vector<uint32_t> image;
    //const bool ownData;

public:
    Image(const std::string& path) {
        int channels; //the value will be ignored as I am forcing 4 channels on load.
        unsigned char* imageData = stbi_load(path.c_str(), &width, &height, &channels, neededChannels);
        assert(imageData != nullptr);
        image = std::vector<uint32_t>((uint32_t*)imageData, (uint32_t*)&imageData[getTotalSizeBytes()]);
        stbi_image_free(imageData);
    }


    // Creates an empty image which can later be filled through the data pointer
    Image(uint32_t width, uint32_t height) :
        width(width),
        height(height)
    {
        image.resize(width * height);
    }

    void save(std::string const& path) {
        stbi_write_png(path.c_str(), width, height, 4, image.data(), getStride());    //stores the image in a png file
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return neededChannels; }
    int getStride() const { return  width * neededChannels; }
    const std::vector<uint32_t>& getData() const { return image; }
    uint32_t* data() { return image.data(); }
    int getTotalSizeBytes() const { return width * height * neededChannels; }

};

//Class used to wrap functionality of the structured buffers used as input/output of the shaders.
class Buffer {
private:
    VkBuffer buf;
    VmaAllocationInfo allocInfo;
    VmaAllocator* allocator;
    VmaAllocation alloc;
    size_t size;
    uint32_t familyIndex;

public:

    void initialize(VmaAllocator* allocator, uint32_t familyIndex){ 
        this->allocator = allocator;
        this->familyIndex = familyIndex;
    }

    VkBuffer& getBuffer() { return buf; }
    size_t getSize() const { return size; }

    void map(void** OutBufferPtr) {
        vmaMapMemory(*allocator, alloc, OutBufferPtr);
    }

    void unmap() {
        vmaUnmapMemory(*allocator, alloc);
    }

    //this function creates a structure buffer and fills the it with the content of a vector of any datatype
    template <typename T>
    VkBuffer& CreateStorage(const T& data) {
        size = data.size() * sizeof(data[0]);
        CreateStorage(size);
        memcpy(allocInfo.pMappedData, data.data(), data.size() * sizeof(data[0]));
        return buf;
    }

    //this template specialization function creates a structure buffer but doesn't initialize its content
    template <>
    VkBuffer& CreateStorage(const size_t& size) {
        this->size = size;
        VkBufferCreateInfo bufCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufCreateInfo.size = size;
        bufCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufCreateInfo.queueFamilyIndexCount = 1;
        bufCreateInfo.pQueueFamilyIndices = &familyIndex;

        VmaAllocationCreateInfo allocCreateInfo = {};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;

        vmaCreateBuffer(*allocator, &bufCreateInfo, &allocCreateInfo, &buf, &alloc, &allocInfo);

        VkMemoryPropertyFlags memPropFlags;
        vmaGetAllocationMemoryProperties(*allocator, alloc, &memPropFlags);
        return buf;
    }

    void destroy() {
        vmaDestroyBuffer(*allocator,buf, alloc);
    }

};

//this class is used to measure the duration of the dispatches in the gpu through timestamps
// the bojects allow to take a maxMeasurements number of measurements before it fills an internal buffer.
// Calling the init method restarts the internal buffer and allows to take maxMeasurements again.
// A measurement starts and stops with the respective methods. The 
class TimestampManager {
private:
    vk::Device device;
    vk::QueryPool queryPoolTimestamps;
    std::vector<uint64_t> timeStamps;
    uint32_t queryIndex;
    uint32_t maxTimeStamps;
    float timeStampPeriod;

public:
    void create(vk::Device& device, vk::PhysicalDevice PhysicalDevice, uint32_t maxMeasurements) {
        this->device = device;
        this->maxTimeStamps = maxMeasurements*2; //each measurement takes 2 timestamps
        
        VkPhysicalDeviceLimits device_limits = PhysicalDevice.getProperties().limits;
        timeStampPeriod = device_limits.timestampPeriod;
        if (timeStampPeriod == 0)
        {
            throw std::runtime_error{ "The selected device does not support timestamp queries!" };
        }

        vk::QueryPoolCreateInfo query_pool_info{};
        query_pool_info.queryType = vk::QueryType::eTimestamp;
        query_pool_info.queryCount = maxTimeStamps;
        queryPoolTimestamps=  device.createQueryPool(query_pool_info);

    }

    void init(vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.resetQueryPool(queryPoolTimestamps, 0, maxTimeStamps);
        queryIndex = 0;
    }

    //start measurement
    void start(vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, queryPoolTimestamps, queryIndex++);       
    }

    //end measurement
    void stop(vk::CommandBuffer& cmdBuffer) {
        cmdBuffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, queryPoolTimestamps, queryIndex++);
    }

    std::vector<uint64_t> retrieveResults() {

        timeStamps.resize(maxTimeStamps);
        device.getQueryPoolResults(queryPoolTimestamps, 0, queryIndex,
            queryIndex * sizeof(uint64_t), timeStamps.data(),
            vk::DeviceSize(sizeof(uint64_t)),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

        std::cout << "---------------------------" << std::endl;
        std::cout.setf(std::ios::fixed, std::ios::floatfield);
        std::cout.precision(3);
        for (uint32_t z = 0; z < queryIndex; z += 2) {
            float delta_in_ms = float(timeStamps[z+1] - timeStamps[z]) * timeStampPeriod / 1'000'000.0f;
            std::cout << "time (" << z/2 << ") = " << delta_in_ms << " ms" << std::endl;
        }

        return timeStamps;
    }

    void destroy() {
        device.destroyQueryPool(queryPoolTimestamps);
    }

};


/*
    Main class with the test code
*/

class GPUImageProcessor{
private:

    void initializeVulkan() {
        vk::ApplicationInfo AppInfo{ "GPU Image Processor", 1, nullptr,0, VK_API_VERSION_1_3 };
        const std::vector<const char*> Layers = { "VK_LAYER_KHRONOS_validation"};

        vk::InstanceCreateInfo InstanceCreateInfo( vk::InstanceCreateFlags(), &AppInfo, Layers, {}); 
        instance = vk::createInstance(InstanceCreateInfo);

        physicalDevice = instance.enumeratePhysicalDevices().front();
        vk::PhysicalDeviceProperties DeviceProps = physicalDevice.getProperties();
        std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
        const uint32_t ApiVersion = DeviceProps.apiVersion;
        std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;

        //Creating the compute queue
        std::vector<vk::QueueFamilyProperties> QueueFamilyProps = physicalDevice.getQueueFamilyProperties();
        auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(), 
            [](const vk::QueueFamilyProperties& Prop) { return Prop.queueFlags & vk::QueueFlagBits::eCompute; });

        computeQueueFamilyIndex = uint32_t(std::distance(QueueFamilyProps.begin(), PropIt));
        std::cout << "Compute Queue Family Index: " << computeQueueFamilyIndex << std::endl;

        //creating the device
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), computeQueueFamilyIndex, 1,&queuePriority);

        //for stamp queries
        vk::PhysicalDeviceHostQueryResetFeatures queryFeatures(true,nullptr);
        vk::PhysicalDeviceSynchronization2Features syncFeatures(true, &queryFeatures);

        vk::DeviceCreateInfo DeviceCreateInfo(vk::DeviceCreateFlags(), DeviceQueueCreateInfo);
        DeviceCreateInfo.setPNext(&syncFeatures);
        device = physicalDevice.createDevice(DeviceCreateInfo);
    }

    void createCommandBuffer() {
        vk::CommandPoolCreateInfo CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), computeQueueFamilyIndex);
        commandPool = device.createCommandPool(CommandPoolCreateInfo);

        vk::CommandBufferAllocateInfo CommandBufferAllocInfo(
            commandPool,                         // Command Pool
            vk::CommandBufferLevel::ePrimary,    // Level
            1);                                  // Num Command Buffers
        const std::vector<vk::CommandBuffer> CmdBuffers = device.allocateCommandBuffers(CommandBufferAllocInfo);
        cmdBuffer = CmdBuffers.front();
    }

    void allocateBuffers(const std::vector<uint32_t>& inputBufferData) {
        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.physicalDevice = physicalDevice;
        allocatorCreateInfo.device = device;
        allocatorCreateInfo.instance = instance;
        vmaCreateAllocator(&allocatorCreateInfo, &allocator);

        inputBufferObject.initialize(&allocator, computeQueueFamilyIndex);
        outputBufferObject.initialize(&allocator, computeQueueFamilyIndex);
        intermediateBufferObject.initialize(&allocator, computeQueueFamilyIndex);

        inputBufferObject.CreateStorage(inputBufferData);
        outputBufferObject.CreateStorage(inputBufferData.size() * sizeof(inputBufferData[0]));
        intermediateBufferObject.CreateStorage(inputBufferData.size() * sizeof(inputBufferData[0]));
    }

    void createLayoutAndDescriptoSet() {
        //layout
        const std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
        };
        vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            DescriptorSetLayoutBinding);
        descriptorSetLayout = device.createDescriptorSetLayout(DescriptorSetLayoutCreateInfo);


        //pipeline layout
        vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), descriptorSetLayout);

        vk::PushConstantRange push_constant;
        push_constant.offset = 0;
        push_constant.size = sizeof(imageData_t);
        push_constant.stageFlags = vk::ShaderStageFlagBits::eCompute;

        PipelineLayoutCreateInfo.pPushConstantRanges = &push_constant;
        PipelineLayoutCreateInfo.pushConstantRangeCount = 1;

        pipelineLayout = device.createPipelineLayout(PipelineLayoutCreateInfo);
        pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());

        //descriptor set
        vk::DescriptorPoolSize DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3);
        std::vector<vk::DescriptorPoolSize> argumentPoolsize{ DescriptorPoolSize ,DescriptorPoolSize ,DescriptorPoolSize };

        vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 3, 3, argumentPoolsize.data());
        descriptorPool = device.createDescriptorPool(DescriptorPoolCreateInfo);

        std::vector<vk::DescriptorSetLayout> argumentDescSetLayout{ descriptorSetLayout ,descriptorSetLayout ,descriptorSetLayout };
        vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo(descriptorPool, 3, argumentDescSetLayout.data());
        descriptorSets = device.allocateDescriptorSets(DescriptorSetAllocInfo);

        vk::DescriptorBufferInfo InBufferInfo(inputBufferObject.getBuffer(), 0, inputBufferObject.getSize());
        vk::DescriptorBufferInfo OutBufferInfo(outputBufferObject.getBuffer(), 0, outputBufferObject.getSize());
        vk::DescriptorBufferInfo IntermediateBufferInfo(intermediateBufferObject.getBuffer(), 0, intermediateBufferObject.getSize());

        //there are two types of effects. The first effect is a single dispatch with an input structured buffer and an output structured buffer.
        //the second effect is made of two dispatches, the first dispatch consumed the input structured buffer, saves the output in an intermediate
        //structured buffer and the second dispatch reads that intermediate structured buffer and saves the output into the output buffer.
        const std::vector<vk::WriteDescriptorSet> WriteDescriptorSets = {
            {descriptorSets[INPUT_OUTPUT], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferInfo},
            {descriptorSets[INPUT_OUTPUT], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo},

            {descriptorSets[INPUT_INTERMEDIATE], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &InBufferInfo},
            {descriptorSets[INPUT_INTERMEDIATE], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &IntermediateBufferInfo},

            {descriptorSets[INTERMEDIATE_OUTPUT], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &IntermediateBufferInfo},
            {descriptorSets[INTERMEDIATE_OUTPUT], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &OutBufferInfo},
        };
        device.updateDescriptorSets(WriteDescriptorSets, {});
    }

    vk::Pipeline createPipeline(std::string const& shader_path) {
        //Create shader module
        std::vector<char> ShaderContents;
        if (std::ifstream ShaderFile{ shader_path.c_str(), std::ios::binary | std::ios::ate })
        {
            const size_t FileSize = ShaderFile.tellg();
            ShaderFile.seekg(0);
            ShaderContents.resize(FileSize, '\0');
            ShaderFile.read(ShaderContents.data(), FileSize);
        }

        vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),                                // Flags
            ShaderContents.size(),                                        // Code size
            reinterpret_cast<const uint32_t*>(ShaderContents.data()));    // Code
        vk::ShaderModule ShaderModule;
        ShaderModule = device.createShaderModule(ShaderModuleCreateInfo);

        vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo(
            vk::PipelineShaderStageCreateFlags(),  // Flags
            vk::ShaderStageFlagBits::eCompute,     // Stage
            ShaderModule,                          // Shader Module
            "Main");                               // Shader Entry Point
        vk::ComputePipelineCreateInfo ComputePipelineCreateInfo(
            vk::PipelineCreateFlags(),    // Flags
            PipelineShaderCreateInfo,     // Shader Create Info struct
            pipelineLayout);              // Pipeline Layout
        auto pipeline = device.createComputePipeline(pipelineCache, ComputePipelineCreateInfo).value;
        device.destroyShaderModule(ShaderModule);
        return pipeline;
    }


public:
    void initialize(const std::vector<std::string>& shaders, const Image& inputImage) {
        width = inputImage.getWidth();
        height = inputImage.getHeight();
        
        //the workgroup size of all the shaders is hardcoded to 256
        dispatch_size_x = uint32_t(std::ceil(float(width * height) / 256.0f));
        dispatch_size_y = 1;

        initializeVulkan();
        timeStamp.create(device, physicalDevice,2);
        allocateBuffers(inputImage.getData());
        createLayoutAndDescriptoSet();
        createCommandBuffer();       
        
        for (const auto& entry : shaders) {
            computePipeline[entry] = createPipeline(entry);
        }
    }

    

    void execute(const std::string& pipeline) {
        vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdBuffer.begin(CmdBufferBeginInfo);
        timeStamp.init(cmdBuffer);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline[pipeline]);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,    // Bind point
            pipelineLayout,                  // Pipeline Layout
            0,                               // First descriptor set
            { descriptorSets[INPUT_OUTPUT] },               // List of descriptor sets
            {});                             // Dynamic offsets

        imageData_t data({ width, height });
        cmdBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(imageData_t), &data);

        timeStamp.start(cmdBuffer);
        cmdBuffer.dispatch(dispatch_size_x,dispatch_size_y, 1);
        timeStamp.stop(cmdBuffer);
        cmdBuffer.end();


        vk::Queue Queue = device.getQueue(computeQueueFamilyIndex, 0);
        vk::Fence Fence = device.createFence(vk::FenceCreateInfo());

        vk::SubmitInfo SubmitInfo(0,                // Num Wait Semaphores
            nullptr,        // Wait Semaphores
            nullptr,        // Pipeline Stage Flags
            1,              // Num Command Buffers
            &cmdBuffer);    // List of command buffers
        Queue.submit({ SubmitInfo }, Fence);
        device.waitForFences({ Fence },             // List of fences
            true,               // Wait All
            uint64_t(-1));      // Timeout

        device.resetCommandPool(commandPool, vk::CommandPoolResetFlags());
        device.destroyFence(Fence);
        timeStamp.retrieveResults();
    }

    void execute(const std::string& pipeline1, const std::string& pipeline2) {
        vk::CommandBufferBeginInfo CmdBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        cmdBuffer.begin(CmdBufferBeginInfo);

        timeStamp.init(cmdBuffer);


        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline[pipeline1]);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,    // Bind point
            pipelineLayout,                  // Pipeline Layout
            0,                               // First descriptor set
            { descriptorSets[INPUT_INTERMEDIATE] },               // List of descriptor sets
            {});                             // Dynamic offsets


        imageData_t data({ width, height });
        cmdBuffer.pushConstants(pipelineLayout,vk::ShaderStageFlagBits::eCompute,0,sizeof(imageData_t), &data);
        timeStamp.start(cmdBuffer);
        cmdBuffer.dispatch(dispatch_size_x, dispatch_size_y, 1);
        timeStamp.stop(cmdBuffer);

       //barrier for the first compute pass
        vk::MemoryBarrier2 memoryBarrier;
        memoryBarrier.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
        .setSrcAccessMask(vk::AccessFlagBits2::eShaderWrite)
        .setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
        .setDstAccessMask(vk::AccessFlagBits2::eShaderRead );
          
        vk::DependencyInfoKHR dependencyInfo;
        dependencyInfo.setMemoryBarrierCount(1)
        .setPMemoryBarriers(&memoryBarrier)
        .setDependencyFlags(vk::DependencyFlagBits::eByRegion);
        
        cmdBuffer.pipelineBarrier2(dependencyInfo);

        //second dispatch
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline[pipeline2]);
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,    // Bind point
            pipelineLayout,                  // Pipeline Layout
            0,                               // First descriptor set
            { descriptorSets[INTERMEDIATE_OUTPUT] },               // List of descriptor sets
            {});

        cmdBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(imageData_t), &data);
        timeStamp.start(cmdBuffer);
        cmdBuffer.dispatch(dispatch_size_x, dispatch_size_y, 1);
        timeStamp.stop(cmdBuffer);

        cmdBuffer.end();

        vk::Queue queue = device.getQueue(computeQueueFamilyIndex, 0);
        vk::Fence fence = device.createFence(vk::FenceCreateInfo());

        vk::SubmitInfo SubmitInfo(0,                // Num Wait Semaphores
            nullptr,        // Wait Semaphores
            nullptr,        // Pipeline Stage Flags
            1,              // Num Command Buffers
            &cmdBuffer);    // List of command buffers
        queue.submit({ SubmitInfo }, fence);
        device.waitForFences({ fence },             // List of fences
            true,               // Wait All
            uint64_t(-1));      // Timeout

        device.resetCommandPool(commandPool, vk::CommandPoolResetFlags());
        device.destroyFence(fence);

        timeStamp.retrieveResults();
    }

    
    void extractOutput(Image & outputImage) {
        int32_t* OutBufferPtr;
        outputBufferObject.map((void**)&OutBufferPtr);
        memcpy(outputImage.data(), OutBufferPtr, outputImage.getTotalSizeBytes());
        outputBufferObject.unmap();
    }


    void destroyVulkan() {
        timeStamp.destroy();
        device.resetCommandPool(commandPool, vk::CommandPoolResetFlags());
        device.destroyDescriptorSetLayout(descriptorSetLayout);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyPipelineCache(pipelineCache);
        
        for (auto& cp : computePipeline) {
            device.destroyPipeline(cp.second);
        }

        device.destroyDescriptorPool(descriptorPool);
        device.destroyCommandPool(commandPool);
        inputBufferObject.destroy();
        outputBufferObject.destroy();
        intermediateBufferObject.destroy();
        vmaDestroyAllocator(allocator);
        device.destroy();
        instance.destroy();
    }

private:
    vk::Instance instance;
    vk::Device device;
    VmaAllocator allocator;
    Buffer inputBufferObject;
    Buffer intermediateBufferObject;
    Buffer outputBufferObject;
    vk::DescriptorPool descriptorPool;
    std::map<std::string, vk::Pipeline> computePipeline;
    vk::CommandPool commandPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::PipelineCache pipelineCache;
    
    vk::CommandBuffer cmdBuffer;
    
    std::vector< vk::DescriptorSet> descriptorSets;
    uint32_t computeQueueFamilyIndex;
    uint32_t dispatch_size_x, dispatch_size_y;
    vk::PhysicalDevice physicalDevice;

    TimestampManager timeStamp;

    uint32_t width, height; //size of the input texture

};

#ifdef CAPTURE_RENDERDOC
//this wrapper class contains functionality to capture render doc captures from code
class RenderDoc {
    RENDERDOC_API_1_4_1* rdocApi;

public:
    RenderDoc() {
        rdocApi = nullptr;
        if (HINSTANCE mod = GetModuleHandleA("renderdoc.dll"))
        {
            std::cout << "loaded library" << std::endl;
            pRENDERDOC_GetAPI RENDERDOC_GetAPI =
                (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_1, (void**)&rdocApi);
            assert(ret == 1);
        }
    }

    void start() {
        if (rdocApi) {
            rdocApi->StartFrameCapture(NULL, NULL);
            std::cout << "start capture" << std::endl;
        }
    }

    void end() {
        if (rdocApi) {
            rdocApi->EndFrameCapture(NULL, NULL);
            std::cout << "end capture" << std::endl;
        }
    }
};
#endif



int main()
{

#ifdef CAPTURE_RENDERDOC
    RenderDoc rdc;
#endif

    const std::vector<std::string> shaders{ "BlackAndWhite.spv", "Gray.spv", "Laplacian.spv", "GaussH.spv", "GaussV.spv" };

    //loads the image
    Image inputImage("dog.jpg");
    Image outputImage(inputImage.getWidth(), inputImage.getHeight());

    GPUImageProcessor test;
    test.initialize(shaders, inputImage);

    //applies shader one
    test.execute(shaders[0]); //executes the shader in the input image
    test.extractOutput(outputImage);   //saves output of the effect into imageout
    outputImage.save("BlackAndWhite.png");
     
    //applies shader two
    test.execute(shaders[1]);
    test.extractOutput(outputImage);
    outputImage.save("Gray.png");

    //applies shader three
    test.execute(shaders[2]);
    test.extractOutput(outputImage);
    outputImage.save("Laplacian.png");

    //applies shader 4 and 5. which is a separable filter, smoothing first horizontally and later vertically
    //I include also an example of how to capture a dispatch in render doc which was useful for me during the design of the shaders
#ifdef CAPTURE_RENDERDOC
    rdc.start();
#endif
    test.execute(shaders[3], shaders[4]);
#ifdef CAPTURE_RENDERDOC
    rdc.end();
#endif
    test.extractOutput(outputImage);
    outputImage.save("Gaussian.png");

    //destroys the objects and releases memory
    test.destroyVulkan();
   
}
