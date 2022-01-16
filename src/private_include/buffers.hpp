#pragma once
#include "vkc/vkc.hpp"

#include <algorithm>
#include <limits>
#include <vulkan/vulkan.hpp>

namespace vkc {
namespace detail {
constexpr vk::BufferUsageFlags staging_usage_flags
		= vk::BufferUsageFlagBits::eTransferSrc
		| vk::BufferUsageFlagBits::eTransferDst;

constexpr vk::MemoryPropertyFlags staging_mem_flags
		= vk::MemoryPropertyFlagBits::eHostVisible
		| vk::MemoryPropertyFlagBits::eHostCoherent;

constexpr vk::BufferUsageFlags gpu_usage_flags
		= vk::BufferUsageFlagBits::eTransferDst
		| vk::BufferUsageFlagBits::eTransferSrc
		| vk::BufferUsageFlagBits::eStorageBuffer;

constexpr vk::MemoryPropertyFlags gpu_mem_flags
		= vk::MemoryPropertyFlagBits::eDeviceLocal;

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

void make_copy_cmd(const vk::Buffer& src, const vk::Buffer& dst,
		size_t byte_size, vk::CommandBuffer& cmd_buf) {
	vk::CommandBufferBeginInfo begin_info{};
	cmd_buf.begin(begin_info);

	vk::BufferCopy copy_region{
		0,
		0,
		byte_size,
	};
	cmd_buf.copyBuffer(src, dst, 1, &copy_region);
	cmd_buf.end();
}
} // namespace detail

// For readability.
using set_id_t = uint32_t;
using binding_id_t = uint32_t;

// Stores shader ids associated with a buffer.
struct buffer_ids {
	// DescriptorSet id.
	set_id_t set_id = (std::numeric_limits<set_id_t>::max)();

	// Binding id.
	binding_id_t binding_id = (std::numeric_limits<binding_id_t>::max)();
};

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

	void bind(const vkc& vkc_inst, vk::DescriptorSet target_desc_set) const {
		if (!has_binding() || !has_set()) {
			throw std::runtime_error{ std::string{ __FUNCTION__ }
				+ " : Trying to bind a buffer without set_id or binding_id." };
		}

		// Specify the buffer to bind to the descriptor.
		vk::DescriptorBufferInfo descriptor_buffer_info{
			_buf.get(),
			0,
			_byte_size,
		};

		vk::WriteDescriptorSet write_descriptor_set{
			target_desc_set, // write to this descriptor set.
			binding_id(), // write to the binding.
			{}, //??
			1, // update a single descriptor.
			vk::DescriptorType::eStorageBuffer,
			nullptr,
			&descriptor_buffer_info,
		};

		// perform the update of the descriptor set.
		vkc_inst.device().updateDescriptorSets(
				1, &write_descriptor_set, 0, nullptr);
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
		return _ids.set_id != (std::numeric_limits<uint32_t>::max)();
	}
	uint32_t set_id() const {
		return _ids.set_id;
	}

	bool has_binding() const {
		return _ids.binding_id != (std::numeric_limits<uint32_t>::max)();
	}
	uint32_t binding_id() const {
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

	// The buffer.
	vk::UniqueBuffer _buf;

	// The memory that backs the buffer.
	vk::UniqueDeviceMemory _mem;
};

// This "buffer" contains 2 buffers.
// One cpu-visible staging buffer, and one gpu-only data buffer.
// Use this to transfer memory to/from the gpu.
struct transfer_buffer {
	transfer_buffer() = default;

	// Create a bound transfer_buffer but doesn't allocate memory.
	transfer_buffer(buffer_ids ids)
			: _staging_buf(
					detail::staging_usage_flags, detail::staging_mem_flags)
			, _gpu_buf(ids, detail::gpu_usage_flags, detail::gpu_mem_flags) {
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
	}

	// Create a bound transfer_buffer and allocate memory.
	transfer_buffer(const vkc& vkc_inst, buffer_ids gpu_ids, size_t byte_size)
			: _staging_buf(vkc_inst, byte_size, detail::staging_usage_flags,
					detail::staging_mem_flags)
			, _gpu_buf(vkc_inst, gpu_ids, byte_size, detail::gpu_usage_flags,
					  detail::gpu_mem_flags) {
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
	}

	// Move-only

	transfer_buffer(transfer_buffer&&) = default;
	transfer_buffer& operator=(transfer_buffer&&) = default;
	transfer_buffer(const transfer_buffer&) = delete;
	transfer_buffer& operator=(const transfer_buffer&) = delete;

	// Operations
	void clear() {
		_staging_buf.clear();
		_gpu_buf.clear();
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
	}

	void resize(const vkc& vkc_inst, size_t byte_size) {
		_staging_buf.resize(vkc_inst, byte_size);
		_gpu_buf.resize(vkc_inst, byte_size);
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
	}

	void bind(const vkc& vkc_inst, vk::DescriptorSet target_desc_set) const {
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
		_gpu_buf.bind(vkc_inst, target_desc_set);
	}

	void make_push_cmd(vk::CommandBuffer&& cmd_buf) {
		if (has_push_cmd()) {
			// Has already been created.
			// TODO : check that byte_size hasn't changed.
			return;
		}

		_push_cmd = std::move(cmd_buf);
		detail::make_copy_cmd(
				_staging_buf.get(), _gpu_buf.get(), byte_size(), _push_cmd);
	}

	void make_pull_cmd(vk::CommandBuffer&& cmd_buf) {
		if (has_pull_cmd()) {
			// Has already been created.
			// TODO : check that byte_size hasn't changed.
			return;
		}

		_pull_cmd = std::move(cmd_buf);
		detail::make_copy_cmd(
				_gpu_buf.get(), _staging_buf.get(), byte_size(), _pull_cmd);
	}

	void push(vkc& vkc_inst, const uint8_t* in_mem) {
		// Map the buffer memory, so that we can read from it on the CPU.
		void* mapped_memory = vkc_inst.device().mapMemory(
				_staging_buf.get_memory(), 0, byte_size());

		uint8_t* out_mem = reinterpret_cast<uint8_t*>(mapped_memory);
		std::copy(in_mem, in_mem + byte_size(), out_mem);

		// Done writing, so unmap.
		vkc_inst.device().unmapMemory(_staging_buf.get_memory());

		// Now, copy the staging buffer to gpu memory.
		vk::SubmitInfo submit_info{
			{},
			nullptr,
			nullptr,
			1, // submit a single command buffer
			// the command buffer to submit.
			&_push_cmd,
		};

		vkc_inst.queue().submit(1, &submit_info, {});
		vkc_inst.queue().waitIdle();
	}

	void pull(vkc& vkc_inst, uint8_t* out_mem) {
		// First, copy the gpu buffer to staging buffer.
		vk::SubmitInfo submit_info{
			{},
			nullptr,
			nullptr,
			1,
			&_pull_cmd,
		};

		vkc_inst.queue().waitIdle();
		vkc_inst.queue().submit(1, &submit_info, {});
		vkc_inst.queue().waitIdle();

		// Map the buffer memory, so that we can read from it on the CPU.
		const void* mapped_memory = vkc_inst.device().mapMemory(
				_staging_buf.get_memory(), 0, byte_size());

		const uint8_t* in_mem = reinterpret_cast<const uint8_t*>(mapped_memory);
		std::copy(in_mem, in_mem + byte_size(), out_mem);

		// Done reading, so unmap.
		vkc_inst.device().unmapMemory(_staging_buf.get_memory());
	}

	// Getters and setters

	size_t byte_size() const {
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
		return _gpu_buf.byte_size();
	}

	size_t capacity() const {
		assert(_staging_buf.capacity() == _gpu_buf.capacity());
		return _gpu_buf.capacity();
	}

	const raw_buffer& staging_buf() const {
		return _staging_buf;
	}
	raw_buffer& staging_buf() {
		return _staging_buf;
	}

	const raw_buffer& gpu_buf() const {
		return _gpu_buf;
	}
	raw_buffer& gpu_buf() {
		return _gpu_buf;
	}

	bool has_push_cmd() const {
		return _push_cmd != vk::CommandBuffer{};
	}
	bool has_pull_cmd() const {
		return _pull_cmd != vk::CommandBuffer{};
	}

private:
	// The staging buffer, accessible from cpu.
	raw_buffer _staging_buf;

	// The actual gpu buffer, not accessible from cpu.
	raw_buffer _gpu_buf;

	// The command to copy from staging to gpu.
	vk::CommandBuffer _push_cmd;

	// The command to copy from gpu to staging.
	vk::CommandBuffer _pull_cmd;
};

// TODO : Allocate and create multiple commands at once, thread.
void make_push_cmds(const vkc& vkc_inst, vk::CommandPool command_pool,
		transfer_buffer& buf) {
	if (buf.has_push_cmd()) {
		return;
	}

	// We are only creating 1 new command buffer. For now.
	vk::CommandBufferAllocateInfo alloc_info{
		command_pool,
		vk::CommandBufferLevel::ePrimary,
		1,
	};

	std::vector<vk::CommandBuffer> new_buf
			= vkc_inst.device().allocateCommandBuffers(alloc_info);
	assert(new_buf.size() == 1);

	buf.make_push_cmd(std::move(new_buf.back()));
}

// TODO : Allocate and create multiple commands at once, thread.
void make_pull_cmds(const vkc& vkc_inst, vk::CommandPool command_pool,
		transfer_buffer& buf) {
	if (buf.has_pull_cmd()) {
		return;
	}

	// We are only creating 1 new command buffer. For now.
	vk::CommandBufferAllocateInfo alloc_info{
		command_pool,
		vk::CommandBufferLevel::ePrimary,
		1,
	};

	std::vector<vk::CommandBuffer> new_buf
			= vkc_inst.device().allocateCommandBuffers(alloc_info);
	assert(new_buf.size() == 1);

	buf.make_pull_cmd(std::move(new_buf.back()));
}
} // namespace vkc
