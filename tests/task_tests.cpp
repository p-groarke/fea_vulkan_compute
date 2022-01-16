#include <gtest/gtest.h>
#include <numeric>
#include <vkc/vulkan_compute.hpp>


extern std::wstring exe_dir();

namespace {
struct p_constants {
	uint32_t test_num = 0;
	float mul = 0.f;
	// float f2 = 0.f;
	// float f3 = 0.f;
};

TEST(task, basics) {
	std::wstring shader_path = exe_dir() + L"data/shaders/task_tests.comp.spv";

	std::vector<float> sent_data(100);
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
		t.push_buffer("buf", sent_data);
		t.record();
		t.submit();
		t.pull_buffer("buf", &recieved_data);

		EXPECT_EQ(sent_data.size(), recieved_data.size());

		for (size_t i = 0; i < recieved_data.size(); ++i) {
			EXPECT_EQ(sent_data[i], recieved_data[i]);
		}

		// pull a second time
		recieved_data.clear();
		t.pull_buffer("buf", &recieved_data);
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
		t.push_buffer("buf", sent_data);
		t.record();
		t.submit();
		t.pull_buffer("buf", &recieved_data);

		EXPECT_EQ(sent_data.size(), recieved_data.size());

		for (size_t i = 0; i < recieved_data.size(); ++i) {
			EXPECT_EQ(sent_data[i] * constants.mul, recieved_data[i]);
		}
	}
}

} // namespace
