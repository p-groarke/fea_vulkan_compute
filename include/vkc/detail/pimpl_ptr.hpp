#pragma once
#include <memory>
#include <utility> // std::forward

namespace vkc {
template <class T, class Deleter = std::default_delete<T>>
struct pimpl_ptr {
	using pointer = T*;
	using element_type = T;
	using deleter_type = Deleter;

	pimpl_ptr()
			: _impl(std::make_unique<T>()) {
	}

	// emplace ctor
	template <class... Args>
	pimpl_ptr(Args&&... args)
			: _impl(std::make_unique<T>(std::forward<Args>(args)...)) {
	}

	pimpl_ptr(const pimpl_ptr& other)
			: _impl(std::make_unique<T>(*other._impl)) {
	}
	pimpl_ptr(pimpl_ptr&& other)
			: _impl(std::make_unique<T>(std::move(*other._impl))) {
		*other._impl = {};
	}
	pimpl_ptr& operator=(const pimpl_ptr& other) {
		if (this != &other) {
			*_impl = *other._impl;
		}
		return *this;
	}
	pimpl_ptr& operator=(pimpl_ptr&& other) {
		if (this != &other) {
			*_impl = std::move(*other._impl);
			*other._impl = {};
		}
		return *this;
	}

protected:
	const std::unique_ptr<element_type, deleter_type> _impl;
};

//// Use this version if you have ambiguous conflicts in your inheritance graph.
// template <class T, class Deleter = std::default_delete<T>>
// struct PimplPtrObj {
//	using pointer = T*;
//	using element_type = T;
//	using deleter_type = Deleter;
//
//	PimplPtrObj()
//			: mImpl(std::make_unique<T>()) {
//	}
//
//	// emplace ctor
//	template <class... Args>
//	PimplPtrObj(Args&&... args)
//			: mImpl(std::make_unique<T>(std::forward<Args>(args)...)) {
//	}
//
//	PimplPtrObj(const PimplPtrObj& other)
//			: mImpl(std::make_unique<T>(*other.mImpl)) {
//	}
//	PimplPtrObj(PimplPtrObj&& other)
//			: mImpl(std::make_unique<T>(std::move(*other.mImpl))) {
//		*other.mImpl = {};
//	}
//	PimplPtrObj& operator=(const PimplPtrObj& other) {
//		if (this != &other) {
//			*mImpl = *other.mImpl;
//		}
//		return *this;
//	}
//	PimplPtrObj& operator=(PimplPtrObj&& other) {
//		if (this != &other) {
//			*mImpl = std::move(*other.mImpl);
//			*other.mImpl = {};
//		}
//		return *this;
//	}
//
//	T& operator*() const {
//		return *mImpl;
//	}
//	T* operator->() const noexcept {
//		return mImpl.get();
//	}
//
// private:
//	const std::unique_ptr<element_type, deleter_type> mImpl;
//};

} // namespace vkc
