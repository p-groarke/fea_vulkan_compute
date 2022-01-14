#include "vkc/vkc.hpp"

#include <filesystem>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vkc {
namespace {
#if defined(NDEBUG)
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_messenger_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
		VkDebugUtilsMessageTypeFlagsEXT /*messageTypes*/,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* /*pUserData*/) {

	printf("Debug Message:\n\t%s : %s\n", pCallbackData->pMessageIdName,
			pCallbackData->pMessage);
	return VK_FALSE;
}
} // namespace

namespace detail {
struct vkc_impl {
	/*
	In order to use Vulkan, you must create an instance.
	*/
	std::vector<const char*> enabled_layers;
	std::vector<const char*> enabled_extensions;

	vk::UniqueInstance instance;

	VkDebugUtilsMessengerEXT debug_utils_messenger;

	/*
	The physical device is some device on the system that supports usage of
	Vulkan. Often, it is simply a graphics card that supports Vulkan.
	*/
	vk::PhysicalDevice physical_device;

	/*
	Then we have the logical device VkDevice, which basically allows
	us to interact with the physical device.
	*/
	vk::UniqueDevice device;

	/*
	In order to execute commands on a device(GPU), the commands must be
	submitted to a queue. The commands are stored in a command buffer, and this
	command buffer is given to the queue. There will be different kinds of
	queues on the device. Not all queues support graphics operations, for
	instance. For this application, we at least want a queue that supports
	compute operations.
	*/
	vk::Queue queue; // a queue supporting compute operations.

	/*
	Groups of queues that have the same capabilities(for instance, they all
	supports graphics and computer operations), are grouped into queue families.

	When submitting a command buffer, you must specify to which queue in the
	family you are submitting to. This variable keeps track of the index of that
	queue in its family.
	*/
	uint32_t queue_family_idx;
};
} // namespace detail

vkc::vkc() {
	/*
	By enabling validation layers, Vulkan will emit warnings if the API
	is used incorrectly. We shall enable the layer
	VK_LAYER_LUNARG_standard_validation, which is basically a collection of
	several useful validation layers.
	*/
	if constexpr (enable_validation_layers) {
		/*
		We get all supported layers with vkEnumerateInstanceLayerProperties.
		*/
		std::vector<vk::LayerProperties> layer_properties
				= vk::enumerateInstanceLayerProperties();

		/*
		And then we simply check if VK_LAYER_LUNARG_standard_validation is
		among the supported layers.
		*/
		{
			const char* layer_name = "VK_LAYER_LUNARG_standard_validation";
			auto it = std::find_if(layer_properties.begin(),
					layer_properties.end(),
					[&](const vk::LayerProperties& prop) {
						return strcmp(layer_name, prop.layerName) == 0;
					});

			if (it == layer_properties.end()) {
				fprintf(stderr,
						"Layer VK_LAYER_LUNARG_standard_validation not "
						"supported\n");
				std::exit(-1);
			}

			_impl->enabled_layers.push_back(layer_name);
		}

		/*
		We need to enable an extension named
		VK_EXT_DEBUG_UTILS_EXTENSION_NAME, in order to be able to print the
		warnings emitted by the validation layer. So again, we just check if
		the extension is among the supported extensions.
		*/
		{
			std::vector<vk::ExtensionProperties> extension_properties
					= vk::enumerateInstanceExtensionProperties();

			auto it = std::find_if(extension_properties.begin(),
					extension_properties.end(),
					[](const vk::ExtensionProperties& prop) {
						return strcmp(VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
									   prop.extensionName)
								== 0;
					});

			if (it == extension_properties.end()) {
				fprintf(stderr,
						"Extension " VK_EXT_DEBUG_UTILS_EXTENSION_NAME " not "
						"supported\n");
				std::exit(-1);
			}

			_impl->enabled_extensions.push_back(
					VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}
	}

	/*
	 Next, we actually create the instance.
	*/

	/*
	 Contains application info. This is actually not that important.
	 The only real important field is apiVersion.
	*/
	vk::ApplicationInfo application_info{
		"libvulkan_compute",
		0,
		"libvulkan_compute",
		0,
		VK_API_VERSION_1_2,
	};

	vk::InstanceCreateInfo create_info{
		{},
		&application_info,
		unsigned(_impl->enabled_layers.size()),
		_impl->enabled_layers.data(),
		unsigned(_impl->enabled_extensions.size()),
		_impl->enabled_extensions.data(),
	};

	/*
	 Actually create the instance.
	 Having created the instance, we can actually start using vulkan.
	*/
	_impl->instance = vk::createInstanceUnique(create_info);

	/*
	 Register a callback function for the extension
	 VK_EXT_DEBUG_UTILS_EXTENSION_NAME, so that warnings emitted from the
	 validation layer are actually printed.
	*/
	if constexpr (enable_validation_layers) {
		vk::DebugUtilsMessageSeverityFlagsEXT severity_flags
				= vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
				| vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;

		vk::DebugUtilsMessageTypeFlagsEXT type_flags
				= vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
				| vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
				| vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;

		VkDebugUtilsMessengerCreateInfoEXT debug_utils_create_info
				= vk::DebugUtilsMessengerCreateInfoEXT{
					  {},
					  severity_flags,
					  type_flags,
					  debug_utils_messenger_callback,
				  };

		auto CreateDebugUtilsMessenger
				= reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
						_impl->instance->getProcAddr(
								"vkCreateDebugUtilsMessengerEXT"));

		if (CreateDebugUtilsMessenger == nullptr) {
			fprintf(stderr, "Could not load vkCreateDebugUtilsMessengerEXT\n");
			std::exit(-1);
		}

		// Create and register callback.
		VkResult res = CreateDebugUtilsMessenger(_impl->instance.get(),
				&debug_utils_create_info, nullptr,
				&_impl->debug_utils_messenger);

		if (res != VK_SUCCESS) {
			fprintf(stderr, "Failed to create debug report callback.\n");
			exit(-1);
		}
	}


	/*
	Find a physical device that can be used with Vulkan.

	So, first we will list all physical devices on the system with
	vkEnumeratePhysicalDevices .
	*/
	if (_impl->instance->enumeratePhysicalDevices().size() == 0) {
		fprintf(stderr, "Could not find a device with vulkan support.\n");
		std::exit(-1);
	}

	/*
	Next, we choose a device that can be used for our purposes.
	With VkPhysicalDeviceFeatures(), we can retrieve a fine-grained list of
	physical features supported by the device. However, in this demo, we are
	simply launching a simple compute shader, and there are no special physical
	features demanded for this task. With VkPhysicalDeviceProperties(), we can
	obtain a list of physical device properties. Most importantly, we obtain a
	list of physical device limitations. For this application, we launch a
	compute shader, and the maximum size of the workgroups and total number of
	compute shader invocations is limited by the physical device, and we should
	ensure that the limitations named maxComputeWorkGroupCount,
	maxComputeWorkGroupInvocations and maxComputeWorkGroupSize are not exceeded
	by our application.  Moreover, we are using a storage buffer in the compute
	shader, and we should ensure that it is not larger than the device can
	handle, by checking the limitation maxStorageBufferRange. However, in our
	application, the workgroup size and total number of shader invocations is
	relatively small, and the storage buffer is not that large, and thus a vast
	majority of devices will be able to handle it. This can be verified by
	looking at some devices at_ http://vulkan.gpuinfo.org/ Therefore, to keep
	things simple and clean, we will not perform any such checks here, and just
	pick the first physical device in the list. But in a real and serious
	application, those limitations should certainly be taken into account.
	*/

	_impl->physical_device
			= _impl->instance->enumeratePhysicalDevices().front();

	vk::PhysicalDeviceProperties gpu_properties
			= _impl->physical_device.getProperties();
	printf("Selected GPU : '%s'\n", gpu_properties.deviceName);

	// for (const vk::PhysicalDevice& device :
	//		_instance.enumeratePhysicalDevices()) {
	//	// TODO : Find best GPU
	//}

	// get the QueueFamilyProperties of the PhysicalDevice
	std::vector<vk::QueueFamilyProperties> queue_family_properties
			= _impl->physical_device.getQueueFamilyProperties();


	// get the first index into queueFamiliyProperties which supports compute
	auto it = std::find_if(queue_family_properties.begin(),
			queue_family_properties.end(),
			[](const vk::QueueFamilyProperties& qfp) {
				return qfp.queueFlags & vk::QueueFlagBits::eCompute;
				//| vk::QueueFlagBits::eTransfer);
			});

	if (it == queue_family_properties.end()) {
		fprintf(stderr, "Couldn't find queue family that supports compute.\n");
		std::exit(-1);
	}

	_impl->queue_family_idx
			= uint32_t(std::distance(queue_family_properties.begin(), it));
	assert(_impl->queue_family_idx < queue_family_properties.size());


	/*
	We create the logical device.
	When creating the device, we also specify what queues it has.
	*/
	float queue_priority = 0.0f;
	vk::DeviceQueueCreateInfo device_queue_create_info{
		{}, _impl->queue_family_idx,
		1, // one queue in family
		&queue_priority, // one queue, so low priority
	};

	// Specify any desired device features here. We do not need any for this
	// application, though.
	vk::PhysicalDeviceFeatures device_features{};

	/*
	Now we create the logical device. The logical device allows us to interact
	with the physical device.
	*/
	vk::DeviceCreateInfo device_create_info{
		{}, 1,
		&device_queue_create_info, // also specify what queue it has
		// TODO : debug me
		// uint32_t(_enabled_layers.size()), // need to specify validation
		// layers

		//_enabled_layers.data(),
		// uint32_t(_enabled_extensions.size()),
		//_enabled_extensions.data(),
		//&device_features,
	};

	_impl->device
			= _impl->physical_device.createDeviceUnique(device_create_info);

	// Get a handle to the only member of the queue family.
	_impl->queue = _impl->device->getQueue(_impl->queue_family_idx, 0);
}

vkc::~vkc() {
	/*
	Clean up non Unique Resources.
	*/
	if constexpr (enable_validation_layers) {
		auto DestroyDebugUtilsMessenger
				= reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
						_impl->instance->getProcAddr(
								"vkDestroyDebugUtilsMessengerEXT"));

		if (DestroyDebugUtilsMessenger == nullptr) {
			fprintf(stderr, "Could not load vkDestroyDebugUtilsMessengerEXT\n");
			std::exit(-1);
		}

		DestroyDebugUtilsMessenger(
				_impl->instance.get(), _impl->debug_utils_messenger, nullptr);
	}
}

const vk::Instance& vkc::instance() const {
	return _impl->instance.get();
}
vk::Instance& vkc::instance() {
	return _impl->instance.get();
}

const vk::PhysicalDevice& vkc::physical_device() const {
	return _impl->physical_device;
}
vk::PhysicalDevice& vkc::physical_device() {
	return _impl->physical_device;
}

const vk::Device& vkc::device() const {
	return _impl->device.get();
}
vk::Device& vkc::device() {
	return _impl->device.get();
}

const vk::Queue& vkc::queue() const {
	return _impl->queue;
}
vk::Queue& vkc::queue() {
	return _impl->queue;
}

uint32_t vkc::queue_family() const {
	return _impl->queue_family_idx;
}

} // namespace vkc
