#pragma once

namespace vkc {
// For readability & type safety.
using set_id_t = uint32_t;
using binding_id_t = uint32_t;

struct set_id_v {
	constexpr set_id_v() = default;
	constexpr set_id_v(uint32_t id_)
			: id(id_) {
	}

	constexpr bool valid() const {
		return id != (std::numeric_limits<set_id_t>::max)();
	}

	friend constexpr bool operator==(set_id_v lhs, set_id_v rhs);
	friend constexpr bool operator!=(set_id_v lhs, set_id_v rhs);

	set_id_t id = (std::numeric_limits<set_id_t>::max)();
};

constexpr bool operator==(set_id_v lhs, set_id_v rhs) {
	return lhs.id == rhs.id;
}
constexpr bool operator!=(set_id_v lhs, set_id_v rhs) {
	return !(lhs == rhs);
}

struct binding_id_v {
	constexpr binding_id_v() = default;
	constexpr binding_id_v(uint32_t id_)
			: id(id_) {
	}

	constexpr bool valid() const {
		return id != (std::numeric_limits<binding_id_t>::max)();
	}

	friend constexpr bool operator==(binding_id_v lhs, binding_id_v rhs);
	friend constexpr bool operator!=(binding_id_v lhs, binding_id_v rhs);

	binding_id_t id = (std::numeric_limits<binding_id_t>::max)();
};

constexpr bool operator==(binding_id_v lhs, binding_id_v rhs) {
	return lhs.id == rhs.id;
}
constexpr bool operator!=(binding_id_v lhs, binding_id_v rhs) {
	return !(lhs == rhs);
}

// Stores shader ids associated with a buffer.
// Basically a wrapper on set id and binding id,
// which are most often required together.
struct buffer_ids {
	// DescriptorSet id.
	set_id_v set_id;

	// Binding id.
	binding_id_v binding_id;
};

} // namespace vkc
