#include "vkc/task.hpp"
#include "private_include/buffers.hpp"
#include "private_include/reflection.hpp"
#include "vkc/vkc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <fea/maps/unsigned_map.hpp>
#include <fea/utils/file.hpp>
#include <spirv_cross.hpp>
#include <vulkan/vulkan.hpp>

namespace vkc {
// Helper structs.
namespace {
// A memory buffer.
struct buffer_info {
	uint32_t set_id = std::numeric_limits<uint32_t>::max();
	uint32_t binding_id = std::numeric_limits<uint32_t>::max();

	size_t push_cmd_buffer_id = std::numeric_limits<size_t>::max();
	size_t pull_cmd_buffer_id = std::numeric_limits<size_t>::max();

	size_t byte_size = 0;

	// the staging buffer, accessible from cpu
	vk::UniqueBuffer staging_buffer;
	// memory that backs the staging buffer
	vk::UniqueDeviceMemory staging_buffer_memory;

	// the actual gpu buffer, not accessible from cpu
	vk::UniqueBuffer gpu_buffer;
	vk::UniqueDeviceMemory gpu_buffer_memory; // its memory
};

// A push constant (uniform).
struct push_constant_info {
	uint32_t set = std::numeric_limits<uint32_t>::max();
	uint32_t binding = std::numeric_limits<uint32_t>::max();
	size_t offset = 0;
	size_t byte_size = 0;
	const void* constant = nullptr;
};


} // namespace

namespace detail {
struct task_impl {
	task_impl() = default;
	task_impl(vkc* v)
			: vkc_inst(v) {
	}

	const vkc& instance() const {
		return *vkc_inst;
	}
	vkc& instance() {
		return *vkc_inst;
	}

	vkc* vkc_inst = nullptr;

	/*
	Descriptors represent resources in shaders. They allow us to use
	things like uniform buffers, storage buffers and images in GLSL. A
	single descriptor represents a single resource, and several
	descriptors are organized into descriptor sets, which are basically
	just collections of descriptors.
	*/
	std::vector<vk::UniqueDescriptorSetLayout> descriptor_set_layouts;
	vk::UniqueDescriptorPool descriptor_pool;
	std::vector<vk::DescriptorSet> descriptor_sets;

	/*
	The pipeline specifies the pipeline that all graphics and compute commands
	pass though in Vulkan. We will be creating a simple compute pipeline in this
	application.
	*/
	vk::UniqueShaderModule compute_shader_module;
	vk::UniquePipelineLayout pipeline_layout;
	vk::UniquePipeline pipeline;

	/*
	The command buffer is used to record commands, that will be submitted to a
	queue. To allocate such command buffers, we use a command pool.
	*/
	vk::UniqueCommandPool command_pool;
	std::vector<vk::CommandBuffer> command_buffers;

	/*
	The push_constants in the shader.
	*/
	std::vector<vk::PushConstantRange> push_constants_ranges;

	/*
	Our buffers.
	*/
	fea::unsigned_map<set_id_t, transfer_buffer> transfer_buffers;

	/*
	string -> id
	*/
	std::unordered_map<std::string, set_id_t> buffer_name_to_set_id;
	std::unordered_map<std::string, push_constant_info>
			push_constants_name_to_info;

	std::array<uint32_t, 3> workgroupsizes = { 1u, 1u, 1u };

	size_t submit_cmd_buffer_id = std::numeric_limits<size_t>::max();
};
} // namespace detail


// Helper functions.
namespace {
void gather_buffer_descriptorsets(vkc& vkc_inst, detail::task_impl& impl,
		const spirv_cross::Compiler& comp) {

	std::vector<buffer_binding> buffer_bindings = reflect_buffer_bindings(comp);

	// Gathered info to call create once.
	fea::unsigned_map<uint32_t, std::vector<vk::DescriptorSetLayoutBinding>>
			layout_bindings;

	for (const buffer_binding& b : buffer_bindings) {
		// Add empty buffer, ready for future filling.
		buffer_ids ids{ b.set_id, b.binding_id };
		impl.transfer_buffers.insert({ b.set_id, transfer_buffer{ ids } });
		impl.buffer_name_to_set_id[b.name] = b.set_id;

		/*
		 Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to
		 the binding point. This binds to layout(std140, binding = 0) buffer
		 buf in the compute shader.
		*/
		vk::DescriptorSetLayoutBinding descriptor_set_layout_binding{
			b.binding_id,
			vk::DescriptorType::eStorageBuffer,
			1,
			vk::ShaderStageFlagBits::eCompute,
		};
		layout_bindings[b.set_id].push_back(descriptor_set_layout_binding);
	}

	/*
	 Here we specify a descriptor set layout. This allows us to bind our
	 descriptors to resources in the shader.
	*/
	std::vector<vk::DescriptorPoolSize> pool_sizes;
	for (const std::pair<uint32_t, std::vector<vk::DescriptorSetLayoutBinding>>&
					kv : layout_bindings) {

		const std::vector<vk::DescriptorSetLayoutBinding>& bindings = kv.second;
		vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_create_info{
			{},
			uint32_t(bindings.size()),
			bindings.data(),
		};

		// Create the descriptor set layout.
		impl.descriptor_set_layouts.push_back(
				vkc_inst.device().createDescriptorSetLayoutUnique(
						descriptor_set_layout_create_info));


		/*
		 So we will allocate a descriptor set here.
		 But we need to first create a descriptor pool to do that.
		*/
		vk::DescriptorPoolSize descriptor_pool_size{
			vk::DescriptorType::eStorageBuffer,
			uint32_t(bindings.size()),
		};
		pool_sizes.push_back(descriptor_pool_size);
	}


	vk::DescriptorPoolCreateInfo descriptor_pool_create_info{
		{},
		uint32_t(layout_bindings.size()),
		uint32_t(pool_sizes.size()),
		pool_sizes.data(),
	};

	// create descriptor pool.
	impl.descriptor_pool = vkc_inst.device().createDescriptorPoolUnique(
			descriptor_pool_create_info);

	/*
	 With the pool allocated, we can now allocate the descriptor set.
	*/
	std::vector<vk::DescriptorSetLayout> layouts;
	for (const vk::UniqueDescriptorSetLayout& l : impl.descriptor_set_layouts) {
		layouts.push_back(l.get());
	}

	vk::DescriptorSetAllocateInfo descriptor_set_allocate_info{
		impl.descriptor_pool.get(), // pool to allocate from.
		uint32_t(layouts.size()),
		layouts.data(),
	};

	// allocate descriptor set.
	impl.descriptor_sets = vkc_inst.device().allocateDescriptorSets(
			descriptor_set_allocate_info);
}

void gather_uniform_descriptorsets(
		detail::task_impl& impl, const spirv_cross::Compiler& comp) {
	std::vector<uniform_binding> uniform_bindings
			= reflect_uniform_bindings(comp);

	// fea::unsigned_map<uint32_t, std::vector<vk::DescriptorSetLayoutBinding>>
	//		layout_bindings;

	for (const uniform_binding& b : uniform_bindings) {
		impl.push_constants_name_to_info[b.name] = {
			b.set_id,
			b.binding_id,
			b.offset,
			b.size,
			nullptr,
		};

		vk::PushConstantRange push_constant_range{
			vk::ShaderStageFlagBits::eCompute,
			uint32_t(b.offset),
			uint32_t(b.size),
		};
		impl.push_constants_ranges.push_back(push_constant_range);
	}
}
} // namespace

task::~task() = default;
// task::task(const task&) = default;
task::task(task&&) = default;
// task& task::operator=(const task&) = default;
task& task::operator=(task&&) = default;

task::task(vkc& vkc_inst, const wchar_t* shader_path)
		: pimpl_ptr(&vkc_inst) {
	// load shader
	// the code in comp.spv was created by running the command:
	// glslangValidator.exe -V shader.comp
	std::filesystem::path shader_filepath = shader_path;
	if (!std::filesystem::exists(shader_filepath)) {
		throw std::invalid_argument{ std::string{ __FUNCTION__ }
			+ " : Invalid shader path, file not found." };
	}

	std::vector<uint8_t> shader_data;

	if (shader_filepath.extension() != ".spv") {
		throw std::invalid_argument{ std::string{ __FUNCTION__ }
			+ " : Provided shader not '.spv'. Task requires precompiled "
			  "shaders." };
	}

	if (!fea::open_binary_file(shader_filepath, shader_data)) {
		fprintf(stderr, "Couldn't open shader file : '%s'\n",
				shader_filepath.string().c_str());
		throw std::runtime_error{ "Couldn't open shader file." };
	}

	size_t padded_size = size_t(std::ceil(shader_data.size() / 4.0) * 4.0);
	for (size_t i = shader_data.size(); i < padded_size; ++i) {
		shader_data.push_back(0);
	}

	/*
	Use spriv_cross reflection to figure out what descriptor sets, bindings
	and buffers we need.
	*/

	spirv_cross::Compiler comp{
		reinterpret_cast<const uint32_t*>(shader_data.data()), padded_size / 4
	};

	gather_buffer_descriptorsets(vkc_inst, *_impl, comp);
	gather_uniform_descriptorsets(*_impl, comp);

	_impl->workgroupsizes = reflect_workinggroup_sizes(comp);

	/*
	We create a compute pipeline here.
	*/

	/*
	Create a shader module. A shader module basically just
	encapsulates some shader code.
	*/
	vk::ShaderModuleCreateInfo shader_module_create_info{
		{},
		padded_size,
		reinterpret_cast<const uint32_t*>(shader_data.data()),
	};

	_impl->compute_shader_module = vkc_inst.device().createShaderModuleUnique(
			shader_module_create_info);

	/*
	 Now let us actually create the compute pipeline.
	 A compute pipeline is very simple compared to a graphics pipeline.
	 It only consists of a single stage with a compute shader.
	 So first we specify the compute shader stage, and it's entry point(main).
	*/
	vk::PipelineShaderStageCreateInfo shader_stage_create_info{
		{},
		vk::ShaderStageFlagBits::eCompute,
		_impl->compute_shader_module.get(),
		"main",
	};

	/*
	 The pipeline layout allows the pipeline to access descriptor sets.
	 So we just specify the descriptor set layout we created earlier.
	*/
	std::vector<vk::DescriptorSetLayout> layouts;
	for (const auto& l : _impl->descriptor_set_layouts) {
		layouts.push_back(l.get());
	}

	vk::PipelineLayoutCreateInfo pipeline_layout_create_info{
		{},
		uint32_t(layouts.size()),
		layouts.data(),
		uint32_t(_impl->push_constants_ranges.size()),
		_impl->push_constants_ranges.data(),
	};

	_impl->pipeline_layout = vkc_inst.device().createPipelineLayoutUnique(
			pipeline_layout_create_info);

	vk::ComputePipelineCreateInfo pipeline_create_info{
		{},
		shader_stage_create_info,
		_impl->pipeline_layout.get(),
	};

	/*
	 Now, we finally create the compute pipeline.
	*/
	_impl->pipeline = vkc_inst.device().createComputePipelineUnique(
			{}, pipeline_create_info);


	/*
	We are getting closer to the end. In order to send commands to the
	device(GPU), we must first record commands into a command buffer. To
	allocate a command buffer, we must first create a command pool. So let us do
	that.
	*/
	vk::CommandPoolCreateInfo command_pool_create_info{
		{},
		// the queue family of this command pool. All command buffers allocated
		// from this command pool, must be submitted to queues of this family
		// ONLY.
		vkc_inst.queue_family(),
	};

	_impl->command_pool = vkc_inst.device().createCommandPoolUnique(
			command_pool_create_info);


	/*
	Now allocate a command buffer from the command pool.
	*/
	vk::CommandBufferAllocateInfo command_buffer_allocate_info{
		_impl->command_pool.get(), // specify the command pool to allocate from.
		// if the command buffer is primary, it can be directly submitted to
		// queues. A secondary buffer has to be called from some primary command
		// buffer, and cannot be directly submitted to a queue. To keep things
		// simple, we use a primary command buffer.
		vk::CommandBufferLevel::ePrimary,
		1, // allocate a single command buffer.
	};

	std::vector<vk::CommandBuffer> new_buf
			= vkc_inst.device().allocateCommandBuffers(
					command_buffer_allocate_info);

	// We are only creating 1 new command buffer.
	assert(new_buf.size() == 1);
	_impl->submit_cmd_buffer_id = _impl->command_buffers.size();
	_impl->command_buffers.push_back(std::move(new_buf.back()));
}

void task::record(
		size_t width /*= 1u*/, size_t height /*= 1u*/, size_t depth /*= 1u*/) {
	// This records the "main task" of our compute shader and stores it for
	// later submitting.

	vk::CommandBufferBeginInfo begin_info{
		// flags optional
	};

	// start recording commands.
	vk::CommandBuffer& command_buffer
			= _impl->command_buffers[_impl->submit_cmd_buffer_id];
	command_buffer.begin(begin_info);

	/*
	We need to bind a pipeline, AND a descriptor set before we dispatch.
	The validation layer will NOT give warnings if you forget these, so be very
	careful not to forget them.
	*/
	command_buffer.bindPipeline(
			vk::PipelineBindPoint::eCompute, _impl->pipeline.get());
	command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
			_impl->pipeline_layout.get(), 0, 1, &_impl->descriptor_sets.back(),
			0, nullptr);

	for (std::pair<const std::string, push_constant_info>& kv :
			_impl->push_constants_name_to_info) {
		push_constant_info& info = kv.second;
		if (info.constant == nullptr) {
			continue;
		}

		command_buffer.pushConstants(_impl->pipeline_layout.get(),
				vk::ShaderStageFlagBits::eCompute, uint32_t(info.offset),
				uint32_t(info.byte_size), info.constant);

		info.constant = nullptr;
	}

	/*
	 Calling vkCmdDispatch basically starts the compute pipeline, and executes
	 the compute shader. The number of workgroups is specified in the
	 arguments.
	*/
	uint32_t x = uint32_t(std::ceil(width / double(_impl->workgroupsizes[0])));
	uint32_t y = uint32_t(std::ceil(height / double(_impl->workgroupsizes[1])));
	uint32_t z = uint32_t(std::ceil(depth / double(_impl->workgroupsizes[2])));
	command_buffer.dispatch(x, y, z);

	// end recording commands.
	command_buffer.end();
}


void task::submit() {
	/*
	Now we shall finally submit the recorded command buffer to a queue.
	*/

	vk::SubmitInfo submit_info{
		{},
		nullptr,
		nullptr,
		1, // submit a single command buffer
		// the command buffer to submit.
		&_impl->command_buffers[_impl->submit_cmd_buffer_id],
	};

	/*
	We submit the command buffer on the queue, at the same time giving a
	fence.
	*/
	_impl->instance().queue().waitIdle();
	_impl->instance().queue().submit(1, &submit_info, {});
	_impl->instance().queue().waitIdle();
}

void task::push_constant(
		const char* constant_name, const void* val, size_t size) {
	push_constant_info& info
			= _impl->push_constants_name_to_info.at(constant_name);

	if (size != info.byte_size) {
		throw std::invalid_argument{
			__FUNCTION__ " : Mismatch between passed in push_constant size and "
						 "shader size."
		};
	}

	info.constant = val;
}

void task::reserve_buffer(const char* buf_name, size_t byte_size) {
	set_id_t id = _impl->buffer_name_to_set_id.at(buf_name);
	transfer_buffer& buf = _impl->transfer_buffers.at(id);
	assert(buf.gpu_buf().set_id() == id);

	// won't allocate if preallocated
	buf.resize(_impl->instance(), byte_size);
	buf.bind(_impl->instance(), _impl->descriptor_sets[id]);
}

void task::push_buffer(
		const char* buf_name, const uint8_t* in_data, size_t byte_size) {
	set_id_t id = _impl->buffer_name_to_set_id.at(buf_name);
	transfer_buffer& buf = _impl->transfer_buffers.at(id);
	assert(buf.gpu_buf().set_id() == id);

	// won't allocate if preallocated
	buf.resize(_impl->instance(), byte_size);
	buf.bind(_impl->instance(), _impl->descriptor_sets[id]);

	make_push_cmds(_impl->instance(), _impl->command_pool.get(), buf);
	buf.push(_impl->instance(), in_data);
}


size_t task::get_buffer_byte_size(const char* buf_name) const {
	set_id_t id = _impl->buffer_name_to_set_id.at(buf_name);
	const transfer_buffer& buf = _impl->transfer_buffers.at(id);
	assert(buf.gpu_buf().set_id() == id);

	return buf.byte_size();
}

void task::pull_buffer(const char* buf_name, uint8_t* out_data) {
	set_id_t id = _impl->buffer_name_to_set_id.at(buf_name);
	transfer_buffer& buf = _impl->transfer_buffers.at(id);
	assert(buf.gpu_buf().set_id() == id);

	make_pull_cmds(_impl->instance(), _impl->command_pool.get(), buf);

	buf.pull(_impl->instance(), out_data);
}

} // namespace vkc
