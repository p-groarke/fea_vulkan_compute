#pragma once
#include "private_include/ids.hpp"
#include "vkc/vkc.hpp"

#include <vulkan/vulkan.hpp>

namespace vkc {
namespace detail {
// Pass in the gpu instance, the buffer for which this memory will be
// used and your desired memory types flag.
vk::MemoryAllocateInfo find_memory_type(const vkc& vkc_inst,
		const vk::Buffer& buffer, vk::MemoryPropertyFlags desired_mem_flags) {

	/*
	 First, we find the memory requirements for the buffer.
	*/
	vk::MemoryRequirements requirements
			= vkc_inst.device().getBufferMemoryRequirements(buffer);


	vk::PhysicalDeviceMemoryProperties memory_properties
			= vkc_inst.physical_device().getMemoryProperties();

	/*
	How does this search work?
	See the documentation of VkPhysicalDeviceMemoryProperties for a detailed
	description.
	*/
	for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
		if ((requirements.memoryTypeBits & (1 << i))
				&& ((memory_properties.memoryTypes[i].propertyFlags
							& desired_mem_flags)
						== desired_mem_flags)) {

			return vk::MemoryAllocateInfo{ requirements.size, i };
		}
	}

	throw std::runtime_error{
		"vkc::task : Couldn't find required memory type."
	};
}

vk::UniqueBuffer make_unique_buffer(
		const vkc& vkc_inst, size_t byte_size, vk::BufferUsageFlags usage) {
	if (byte_size == 0) {
		return {};
	}

	// Maybe need to expose exclusivity.
	vk::BufferCreateInfo buffer_create_info{
		{}, byte_size, usage,
		vk::SharingMode::eExclusive, // exclusive to a single queue family
	};

	return vkc_inst.device().createBufferUnique(buffer_create_info);
}

vk::UniqueDeviceMemory make_unique_memory(const vkc& vkc_inst,
		const vk::Buffer& buffer, vk::MemoryPropertyFlags mem_flags) {
	if (!buffer) {
		return {};
	}

	vk::MemoryAllocateInfo allocate_info
			= find_memory_type(vkc_inst, buffer, mem_flags);

	// allocate memory on device.
	return vkc_inst.device().allocateMemoryUnique(allocate_info);
}
} // namespace detail


// This is a basic gpu buffer, backed by memory.
struct raw_buffer {
	raw_buffer() = default;

	// Creates a raw buffer associated with buffer_ids without allocating
	// memory.
	raw_buffer(buffer_ids ids)
			: _ids(ids) {
	}

	// Creates a bound buffer and allocates memory.
	raw_buffer(const vkc& vkc_inst, buffer_ids ids, size_t byte_size,
			vk::BufferUsageFlags usage_flags, vk::MemoryPropertyFlags mem_flags)
			: _ids(ids)
			, _usage_flags(usage_flags)
			, _mem_flags(mem_flags)
			, _byte_size(byte_size)
			, _reserved_size(_byte_size)
			, _buf(detail::make_unique_buffer(
					  vkc_inst, _byte_size, _usage_flags))
			, _mem(detail::make_unique_memory(
					  vkc_inst, _buf.get(), _mem_flags)) {

		if (_byte_size == 0) {
			return;
		}

		// Now associate that allocated memory with the buffer. With that, the
		// buffer is backed by actual memory.
		vkc_inst.device().bindBufferMemory(_buf.get(), _mem.get(), 0);
	}

	// Creates a unbound buffer and allocates memory.
	raw_buffer(const vkc& vkc_inst, size_t byte_size,
			vk::BufferUsageFlags usage, vk::MemoryPropertyFlags mem_flags)
			: raw_buffer(vkc_inst, {}, byte_size, usage, mem_flags) {
	}

	// Creates a bound buffer without allocating memory.
	raw_buffer(buffer_ids ids, vk::BufferUsageFlags usage_flags,
			vk::MemoryPropertyFlags mem_flags)
			: _ids(ids)
			, _usage_flags(usage_flags)
			, _mem_flags(mem_flags) {
	}

	// Creates a unbound buffer without allocating memory.
	raw_buffer(
			vk::BufferUsageFlags usage_flags, vk::MemoryPropertyFlags mem_flags)
			: raw_buffer({}, usage_flags, mem_flags) {
	}

	// Move-only.
	raw_buffer(raw_buffer&&) = default;
	raw_buffer& operator=(raw_buffer&&) = default;
	raw_buffer(const raw_buffer&) = delete;
	raw_buffer& operator=(const raw_buffer&) = delete;


	// Operations.

	void clear() {
		_byte_size = 0;
	}

	void resize(const vkc& vkc_inst, size_t new_byte_size) {
		if (new_byte_size <= _reserved_size) {
			_byte_size = new_byte_size;
			return;
		}

		// Growing
		_byte_size = new_byte_size;
		_reserved_size = new_byte_size;

		_buf = detail::make_unique_buffer(
				vkc_inst, new_byte_size, _usage_flags);
		_mem = detail::make_unique_memory(vkc_inst, _buf.get(), _mem_flags);
		vkc_inst.device().bindBufferMemory(_buf.get(), _mem.get(), 0);
	}

	void bind(const vkc& vkc_inst, vk::DescriptorSet target_desc_set) {
		if (!has_binding() || !has_set()) {
			throw std::runtime_error{ std::string{ __FUNCTION__ }
				+ " : Trying to bind a buffer without set_id and binding_id." };
		}

		if (_bound_byte_size == _byte_size) {
			// Already bound to correct size.
			return;
		}

		// Specify the buffer to bind to the descriptor.
		vk::DescriptorBufferInfo descriptor_buffer_info{
			_buf.get(),
			0,
			_byte_size,
		};

		vk::WriteDescriptorSet write_descriptor_set{
			target_desc_set, // write to this descriptor set.
			binding_id().id, // write to the binding.
			{}, //??
			1, // update a single descriptor.
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&descriptor_buffer_info,
		};

		// perform the update of the descriptor set.
		vkc_inst.device().updateDescriptorSets(
				1, &write_descriptor_set, 0, nullptr);

		_bound_byte_size = _byte_size;
	}

	// Getters and setters.

	size_t byte_size() const {
		return _byte_size;
	}

	size_t capacity() const {
		return _reserved_size;
	}

	const vk::Buffer& get() const {
		return _buf.get();
	}
	vk::Buffer& get() {
		return _buf.get();
	}

	const vk::DeviceMemory& get_memory() const {
		return _mem.get();
	}
	vk::DeviceMemory& get_memory() {
		return _mem.get();
	}

	bool has_set() const {
		return _ids.set_id.valid();
	}
	set_id_v set_id() const {
		return _ids.set_id;
	}

	bool has_binding() const {
		return _ids.binding_id.valid();
	}
	binding_id_v binding_id() const {
		return _ids.binding_id;
	}

private:
	// Binding and descriptor set ids. Can be invalid.
	buffer_ids _ids;

	// Store flags for future operations (resize, etc).
	vk::BufferUsageFlags _usage_flags;
	vk::MemoryPropertyFlags _mem_flags;

	// Byte size of buffer.
	size_t _byte_size = 0;

	// Actual size of allocated memory.
	size_t _reserved_size = 0;

	// Byte size when last bound.
	// Used to skip binding if unnecessary.
	size_t _bound_byte_size = 0;

	// The buffer.
	vk::UniqueBuffer _buf;

	// The memory that backs the buffer.
	vk::UniqueDeviceMemory _mem;
};
} // namespace vkc
