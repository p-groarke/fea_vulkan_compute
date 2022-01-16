#include <gtest/gtest.h>
#include <vkc/vkc.hpp>

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

const char* argv0 = nullptr;

std::wstring exe_dir() {
	std::wstring ret;
	wchar_t dir[MAX_PATH];
	GetModuleFileNameW(nullptr, dir, MAX_PATH);
	ret = dir;
	ret = ret.substr(0, ret.find_last_of(L"\\\\") + 1);

	assert(!ret.empty());
	return ret;
}

int main(int argc, char** argv) {
	argv0 = argv[0];

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
