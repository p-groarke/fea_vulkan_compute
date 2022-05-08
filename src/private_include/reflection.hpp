#pragma once
#include "private_include/ids.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <fea/utils/throw.hpp>
#include <spirv_cross.hpp>
#include <vulkan/vulkan.hpp>

namespace vkc {
struct buffer_binding_info {
	buffer_ids ids;
	std::string name;
};

struct uniform_binding_info {
	buffer_ids ids;
	std::string name;
	size_t offset = 0;
	size_t size = 0;
};


std::vector<buffer_binding_info> reflect_buffer_bindings(
		const spirv_cross::Compiler& comp) {
	spirv_cross::ShaderResources resources = comp.get_shader_resources();
	std::vector<buffer_binding_info> ret;
	ret.reserve(resources.storage_buffers.size());

	for (const spirv_cross::Resource& res : resources.storage_buffers) {
		buffer_binding_info b;
		b.ids.set_id
				= comp.get_decoration(res.id, spv::DecorationDescriptorSet);
		b.ids.binding_id = comp.get_decoration(res.id, spv::DecorationBinding);
		b.name = res.name;

		// Notice how we're using type_id here because we need the array
		// information and not decoration information.
		const spirv_cross::SPIRType& type = comp.get_type(res.type_id);

		if (type.array.size() == 0) {
			ret.push_back(std::move(b));
			continue;
		}

		// We have an array of buffers.
		// Spirv-cross orders nested arrays backwards.
		// https://github.com/KhronosGroup/SPIRV-Cross/wiki/Reflection-API-user-guide

		// for (int i)

		size_t dimension = type.array.size();
		size_t count = type.array[0];
		bool lit = type.array_size_literal[0];
		dimension;
		count;
		lit;
		// printf("%zu\n", type.array.size());
		// printf(type.array.size()); // 1, because it's one dimension.
		// printf(type.array[0]); // 10
		// printf(type.array_size_literal[0]); // true
	}

	return ret;
}

std::vector<uniform_binding_info> reflect_uniform_bindings(
		const spirv_cross::Compiler& comp) {
	spirv_cross::ShaderResources resources = comp.get_shader_resources();
	std::vector<uniform_binding_info> ret;
	ret.reserve(resources.push_constant_buffers.size());

	for (const spirv_cross::Resource& res : resources.push_constant_buffers) {
		uniform_binding_info b;
		b.ids.set_id
				= comp.get_decoration(res.id, spv::DecorationDescriptorSet);
		b.ids.binding_id = comp.get_decoration(res.id, spv::DecorationBinding);

		spirv_cross::SmallVector<spirv_cross::BufferRange> ranges
				= comp.get_active_buffer_ranges(res.id);

		if (ranges.empty()) {
			continue;
		}

		b.name = res.name;
		b.offset = ranges.front().offset;
		b.size = 0;

		for (const spirv_cross::BufferRange& range : ranges) {
			b.size += range.range;
			// range.index; // Struct member index
			// range.offset; // Offset into struct
			// range.range; // Size of struct member
		}

		if (b.size > 128) {
			fea::maybe_throw<std::runtime_error>(__FUNCTION__, __LINE__,
					"Vulkan limits the size of push_constants to "
					"128 Bytes. Push_constant struct too big.");
		}

		ret.push_back(std::move(b));
	}

	return ret;
}

std::array<uint32_t, 3> reflect_workinggroup_sizes(
		const spirv_cross::Compiler& comp) {
	std::array<uint32_t, 3> ret{ 1u, 1u, 1u };

	spirv_cross::SpecializationConstant x_unused, y_unused, z_unused;
	uint32_t id = comp.get_work_group_size_specialization_constants(
			x_unused, y_unused, z_unused);

	if (id == 0) {
		fea::maybe_throw<std::runtime_error>(__FUNCTION__, __LINE__,
				"Compute shader must declare work group sizes.");
	}

	const spirv_cross::SPIRConstant& workgroup_vals = comp.get_constant(id);
	size_t count = std::min(uint32_t(ret.size()), workgroup_vals.vector_size());
	assert(count != 0);

	for (size_t i = 0; i < count; ++i) {
		ret[i] = workgroup_vals.vector().r[i].i32;
	}
	return ret;
}

} // namespace vkc
