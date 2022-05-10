#pragma once
#include "private_include/ids.hpp"
#include "private_include/raw_buffer.hpp"
#include "vkc/vkc.hpp"

#include <algorithm>
#include <limits>
#include <vulkan/vulkan.hpp>

namespace fea {
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


// This buffer contains 2 raw_buffers.
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

	void bind(const vkc& vkc_inst, vk::DescriptorSet target_desc_set) {
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
		_gpu_buf.bind(vkc_inst, target_desc_set);
	}

	void make_push_cmd(vk::CommandBuffer&& cmd_buf) {
		if (has_push_cmd()) {
			// Has already been created at correct size.
			return;
		}

		_push_cmd = std::move(cmd_buf);
		detail::make_copy_cmd(
				_staging_buf.get(), _gpu_buf.get(), byte_size(), _push_cmd);
		_push_cmd_byte_size = byte_size();
	}

	void make_pull_cmd(vk::CommandBuffer&& cmd_buf) {
		if (has_pull_cmd()) {
			// Has already been created at correct size.
			return;
		}

		_pull_cmd = std::move(cmd_buf);
		detail::make_copy_cmd(
				_gpu_buf.get(), _staging_buf.get(), byte_size(), _pull_cmd);
		_pull_cmd_byte_size = byte_size();
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

		vk::Result res = vkc_inst.queue().submit(1, &submit_info, {});
		if (res != vk::Result::eSuccess) {
			fprintf(stderr, "Buffer push submit failed with result : '%d'\n",
					res);
			return;
		}

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
		vk::Result res = vkc_inst.queue().submit(1, &submit_info, {});
		if (res != vk::Result::eSuccess) {
			fprintf(stderr, "Buffer pull submit failed with result : '%d'\n",
					res);
			return;
		}

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
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
		return _push_cmd != vk::CommandBuffer{}
		&& _push_cmd_byte_size == _staging_buf.byte_size();
	}
	bool has_pull_cmd() const {
		assert(_staging_buf.byte_size() == _gpu_buf.byte_size());
		return _pull_cmd != vk::CommandBuffer{}
		&& _pull_cmd_byte_size == _staging_buf.byte_size();
	}

private:
	// The staging buffer, accessible from cpu.
	raw_buffer _staging_buf;

	// The actual gpu buffer, not accessible from cpu.
	raw_buffer _gpu_buf;

	// The command to copy from staging to gpu.
	vk::CommandBuffer _push_cmd;

	// The push command byte_size.
	// Used to trigger creation of new command when size has changed.
	size_t _push_cmd_byte_size = 0;

	// The command to copy from gpu to staging.
	vk::CommandBuffer _pull_cmd;

	// The pull command byte_size.
	// Used to trigger creation of new command when size has changed.
	size_t _pull_cmd_byte_size = 0;
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
} // namespace fea
