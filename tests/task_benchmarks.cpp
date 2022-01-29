#include <fea/utils/platform.hpp>
#if defined(FEA_RELEASE) && defined(FEA_VKC_BENCHMARKS)

#include <fea/utils/file.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <tbb/parallel_for.h>
#include <vkc/vulkan_compute.hpp>

extern const char* argv0;

namespace {
struct p_constants {
	uint32_t test_num = 0;
	float mul = 0.f;
	// float f2 = 0.f;
	// float f3 = 0.f;
};

TEST(task, benchmarks) {
	// std::filesystem::path exe_path = fea::executable_dir(argv0);
	// std::filesystem::path shader_path
	//		= exe_path / L"data/shaders/task_tests.comp.spv";

	// std::vector<float> sent_data(100);
	// std::iota(sent_data.begin(), sent_data.end(), 0.f);
	// std::vector<float> recieved_data;

	// p_constants constants;

	// vkc::vkc gpu;
	// vkc::task t{ gpu, shader_path.c_str() };
}
} // namespace
#endif
