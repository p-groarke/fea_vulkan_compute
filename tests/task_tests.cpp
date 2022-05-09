#include <fea/utils/file.hpp>
#include <gtest/gtest.h>
#include <numeric>
#include <tbb/parallel_for.h>
#include <vkc/vulkan_compute.hpp>

extern const char* argv0;

namespace {
namespace vkc = fea::vkc;

struct p_constants {
	uint32_t test_num = 0;
	float mul = 0.f;
	// float f2 = 0.f;
	// float f3 = 0.f;
};

TEST(task, basics) {
	std::filesystem::path exe_path = fea::executable_dir(argv0);
	std::filesystem::path shader_path
			= exe_path / L"data/shaders/task_tests.comp.spv";

	std::vector<float> sent_data(100);
	std::vector<float> empty_vec;
	std::iota(sent_data.begin(), sent_data.end(), 0.f);
	std::vector<float> recieved_data;

	p_constants constants;

	vkc::vkc gpu;
	vkc::task t{ gpu, shader_path.c_str() };

	// Test 0
	// Does nothing.
	{
		constants.test_num = 0;

		t.push_constant("p_constants", constants);
		t.push_buffer("buf1", sent_data);
		// t.push_buffer("buf2", empty_vec);
		t.submit();
		t.pull_buffer("buf1", &recieved_data);

		EXPECT_EQ(sent_data.size(), recieved_data.size());

		for (size_t i = 0; i < recieved_data.size(); ++i) {
			EXPECT_EQ(sent_data[i], recieved_data[i]);
		}

		// pull a second time
		recieved_data.clear();
		t.pull_buffer("buf1", &recieved_data);
		EXPECT_EQ(sent_data.size(), recieved_data.size());
		for (size_t i = 0; i < recieved_data.size(); ++i) {
			EXPECT_EQ(sent_data[i], recieved_data[i]);
		}
	}

	// Test 1
	// Multiplies values by mul.
	{
		recieved_data.clear();
		constants.test_num = 1;
		constants.mul = 2.f;

		t.push_constant("p_constants", constants);
		t.push_buffer("buf1", sent_data);
		t.submit();
		t.pull_buffer("buf1", &recieved_data);

		EXPECT_EQ(sent_data.size(), recieved_data.size());

		for (size_t i = 0; i < recieved_data.size(); ++i) {
			EXPECT_EQ(sent_data[i] * constants.mul, recieved_data[i]);
		}
	}

	// Test 2
	// Blend buffers and pull from out.
	{
		recieved_data.clear();
		constants.test_num = 2;

		t.push_constant("p_constants", constants);
		t.push_buffer("buf1", sent_data);
		t.push_buffer("buf2", sent_data);
		t.reserve_buffer<float>("out_buf", sent_data.size());
		t.submit();
		t.pull_buffer("out_buf", &recieved_data);

		EXPECT_EQ(sent_data.size(), recieved_data.size());

		for (size_t i = 0; i < recieved_data.size(); ++i) {
			float expected = sent_data[i] + sent_data[i];
			EXPECT_EQ(expected, recieved_data[i]);
		}

		// Now resize buffers to test if the commands get rebuilt.
		recieved_data.clear();
		std::vector<float> new_send_data = sent_data;
		new_send_data.insert(
				new_send_data.end(), sent_data.begin(), sent_data.end());
		new_send_data.insert(
				new_send_data.end(), sent_data.begin(), sent_data.end());

		t.push_buffer("buf1", new_send_data);
		t.push_buffer("buf2", new_send_data);
		t.reserve_buffer<float>("out_buf", new_send_data.size());
		t.submit();
		t.pull_buffer("out_buf", &recieved_data);

		EXPECT_EQ(new_send_data.size(), recieved_data.size());

		for (size_t i = 0; i < recieved_data.size(); ++i) {
			float expected = new_send_data[i] + new_send_data[i];
			EXPECT_EQ(expected, recieved_data[i]);
		}
	}
}

//// TODO
// TEST(task, task_level_threading) {
//	constexpr size_t num_tasks = 1'000;
//	std::filesystem::path exe_path = fea::executable_dir(argv0);
//	std::filesystem::path shader_path
//			= exe_path / L"data/shaders/task_tests.comp.spv";
//
//	std::vector<float> sent_data(100);
//	std::iota(sent_data.begin(), sent_data.end(), 0.f);
//
//	vkc::vkc gpu;
//
//	// Use grainsize 1 to force many threads.
//	tbb::parallel_for(tbb::blocked_range<size_t>{ 0, num_tasks, 1 },
//			[&](const tbb::blocked_range<size_t>& range) {
//				for (size_t i = range.begin(); i < range.end(); ++i) {
//					vkc::task t{ gpu, shader_path.c_str() };
//					std::vector<float> recieved_data;
//
//					p_constants constants;
//					constants.test_num = 2;
//
//					t.push_constant("p_constants", constants);
//					t.push_buffer("buf1", sent_data);
//					t.push_buffer("buf2", sent_data);
//					t.reserve_buffer<float>("out_buf", sent_data.size());
//					t.submit();
//					t.pull_buffer("out_buf", &recieved_data);
//
//					EXPECT_EQ(sent_data.size(), recieved_data.size());
//
//					for (size_t j = 0; j < recieved_data.size(); ++j) {
//						float expected = sent_data[j] + sent_data[j];
//						EXPECT_EQ(expected, recieved_data[j]);
//					}
//				}
//			});
//}

} // namespace
