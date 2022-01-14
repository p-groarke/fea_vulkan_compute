#pragma once
#include "vkc/detail/pimpl_ptr.hpp"

namespace spirv_cross {
class Compiler;
} // namespace spirv_cross

namespace vkc {
struct vkc;

namespace detail {
struct task_impl;
}

// A compute task.
// Use this to loads shader, push data, execute shader and pull data.
struct task : pimpl_ptr<detail::task_impl> {
	// Must be precompiled shader ending in .spv
	task(vkc& vkc_inst, const wchar_t* shader_path);

	// Enqueue your push_constant block.
	// constant_name is the name of the block in the shader.
	// Constant must stay valid until the next call to submit.
	template <class T>
	void update_push_constant(const wchar_t* constant_name, const T& val);

	// Size is the number of elements (NOT BYTES).
	// Call this if you never have to push data to the shader.
	// AKA, if your compute shader is purely a data generator.
	template <class T>
	void reserve_buffer(vkc& vkc_inst, const wchar_t* buf_name, size_t size);

	// Copies your data into gpu buffer.
	// If you don't need to use this (you don't copy any data to the gpu), you
	// must call reserve_buffer!
	template <class T>
	void push_buffer(vkc& vkc_inst, const wchar_t* buf_name,
			const std::vector<T>& in_data);

	// This records the submit work to be done. You should call this only once
	// (while you can call submit as many times are required).
	// The provided width, height and depth will be divided by shader work group
	// sizes, to compute the number of group counts.
	void record(size_t width = 1u, size_t height = 1u, size_t depth = 1u);

	// Executes the compute shader. Blocking.
	void submit(vkc& vkc_inst);

	// Copies your gpu buffer into data.
	template <class T>
	void pull_buffer(
			vkc& vkc_inst, const wchar_t* buf_name, std::vector<T>* data);

	// const vk::CommandPool& command_pool() const;
	// vk::CommandPool& command_pool();

	// const std::vector<vk::CommandBuffer>& command_buffers() const;
	// std::vector<vk::CommandBuffer>& command_buffers();

private:
	void init_storage_resources(
			vkc& vkc_inst, const spirv_cross::Compiler& comp);
	void init_uniforms(const spirv_cross::Compiler& comp);

	void allocate_buffer(vkc& vkc_inst, buffer_info& buf_info, size_t buf_size);
	void push_buffer(
			vkc& vkc_inst, buffer_info& buf_info, const char* in_memory);
	void pull_buffer(const vkc& vkc_inst, const buffer_info& buf_info,
			char* out_memory) const;

	size_t record_buffer_transfer(vkc& vkc_inst, size_t byte_size,
			const vk::Buffer& src, const vk::Buffer& dst);
};

template <class T>
void task::update_push_constant(const wchar_t* constant_name, const T& val) {

	push_constant_info& info
			= _impl->push_constants_name_to_info.at(constant_name);
	if (sizeof(T) != info.byte_size) {
		throw std::invalid_argument{
			__FUNCTION__ " : Mismatch between passed in push_constant size and "
						 "shader size."
		};
	}
	info.constant = &val;
}

template <class T>
void task::reserve_buffer(vkc& vkc_inst, const wchar_t* buf_name, size_t size) {
	buffer_info& info = _impl->buffer_name_to_info.at(buf_name);
	size_t byte_size = size * sizeof(T);
	allocate_buffer(vkc_inst, info, byte_size);
}

template <class T>
void task::push_buffer(
		vkc& vkc_inst, const wchar_t* buf_name, const std::vector<T>& in_data) {
	buffer_info& info = _impl->buffer_name_to_info.at(buf_name);
	size_t byte_size = in_data.size() * sizeof(T);

	// won't allocate if preallocated
	allocate_buffer(vkc_inst, info, byte_size);

	if (info.push_cmd_buffer_id == std::numeric_limits<size_t>::max()) {
		info.push_cmd_buffer_id = record_buffer_transfer(vkc_inst, byte_size,
				info.staging_buffer.get(), info.gpu_buffer.get());
	}

	push_buffer(vkc_inst, info, reinterpret_cast<const char*>(in_data.data()));
}

template <class T>
void task::pull_buffer(
		vkc& vkc_inst, const wchar_t* buf_name, std::vector<T>* out_data) {
	buffer_info& info = _impl->buffer_name_to_info.at(buf_name);
	out_data->resize(info.byte_size / sizeof(T));

	if (info.pull_cmd_buffer_id == std::numeric_limits<size_t>::max()) {
		info.pull_cmd_buffer_id
				= record_buffer_transfer(vkc_inst, info.byte_size,
						info.gpu_buffer.get(), info.staging_buffer.get());
	}

	pull_buffer(vkc_inst, info, reinterpret_cast<char*>(out_data->data()));
}
} // namespace vkc
