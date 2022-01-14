#pragma once
#include "vkc/detail/pimpl_ptr.hpp"

#include <cstdint>

namespace vk {
class Instance;
class PhysicalDevice;
class Device;
class Queue;
} // namespace vk

namespace vkc {
namespace detail {
struct vkc_impl;
}

// Initializes vulkan and stores the global state.
struct vkc : pimpl_ptr<detail::vkc_impl> {
	vkc();
	~vkc();

	// void compute();
	// void save_image();

	const vk::Instance& instance() const;
	vk::Instance& instance();

	const vk::PhysicalDevice& physical_device() const;
	vk::PhysicalDevice& physical_device();

	const vk::Device& device() const;
	vk::Device& device();

	const vk::Queue& queue() const;
	vk::Queue& queue();

	uint32_t queue_family() const;
};

} // namespace vkc
