#pragma once
#include "vkc/detail/pimpl_ptr.hpp"

#include <cstdint>
#include <vector> // todo : span

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
	~task();

	task(task&&);
	task& operator=(task&&);

	// Move-only.
	task(const task&) = delete;
	task& operator=(const task&) = delete;

	// Enqueue your push_constant block.
	// constant_name is the name of the block in the shader.
	// Constant must stay valid until the next call to submit.
	template <class T>
	void push_constant(const char* constant_name, const T& val);

	// Size is the number of elements (NOT BYTES).
	// Call this if you never have to push data to the shader.
	// AKA, if your compute shader is purely a data generator.
	template <class T>
	void reserve_buffer(const char* buf_name, size_t size);

	// Copies your data into gpu buffer.
	// If you don't need to use this (you don't copy any data to the gpu), you
	// must call reserve_buffer.
	template <class T>
	void push_buffer(const char* buf_name, const std::vector<T>& in_data);

	// Executes the compute shader.
	// Blocking.
	// Uses working group sizes width = 1, height = 1, depth = 1.
	// The width, height and depth will be divided by shader work group
	// sizes, to compute the number of group counts.
	void submit();

	// Executes the compute shader with provided working group sizes.
	// Blocking.
	// The provided width, height and depth will be divided by shader work group
	// sizes, to compute the number of group counts.
	void submit(size_t width, size_t height, size_t depth);

	// Copies your gpu buffer into data.
	template <class T>
	void pull_buffer(const char* buf_name, std::vector<T>* data);

private:
	void push_constant(
			const char* constant_name, const void* val, size_t byte_size);
	void reserve_buffer(const char* buf_name, size_t byte_size);
	void push_buffer(
			const char* buf_name, const uint8_t* in_data, size_t byte_size);

	size_t get_buffer_byte_size(const char* buf_name) const;
	void pull_buffer(const char* buf_name, uint8_t* out_data);
};


// Template implementations.

template <class T>
void task::push_constant(const char* constant_name, const T& val) {
	push_constant(constant_name, &val, sizeof(T));
}

template <class T>
void task::reserve_buffer(const char* buf_name, size_t size) {
	reserve_buffer(buf_name, sizeof(T) * size);
}

template <class T>
void task::push_buffer(const char* buf_name, const std::vector<T>& in_data) {
	push_buffer(buf_name, reinterpret_cast<const uint8_t*>(in_data.data()),
			sizeof(T) * in_data.size());
}

template <class T>
void task::pull_buffer(const char* buf_name, std::vector<T>* out_data) {
	out_data->resize(get_buffer_byte_size(buf_name) / sizeof(T));
	pull_buffer(buf_name, reinterpret_cast<uint8_t*>(out_data->data()));
}
} // namespace vkc
