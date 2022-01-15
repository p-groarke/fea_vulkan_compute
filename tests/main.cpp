#include <array>
#include <fea/benchmark/benchmark.hpp>
#include <gtest/gtest.h>
#include <vkc/vulkan_compute.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {
const char* argv0 = nullptr;

struct pixel {
	float r = 0.f;
	float g = 0.f;
	float b = 0.f;
	float a = 1.f;
};

struct size_block {
	uint32_t width = 3200;
	uint32_t height = 2400;
};

TEST(vulkan_compute, basics) {
	std::wstring exe_dir;
	{
		wchar_t dir[MAX_PATH];
		GetModuleFileNameW(nullptr, dir, MAX_PATH);
		exe_dir = dir;
		exe_dir = exe_dir.substr(0, exe_dir.find_last_of(L"\\\\") + 1);
	}

	std::wstring shader_path = exe_dir + L"data/shaders/mandelbrot.comp.spv";

	size_block size;
	size_t pixel_size = size.width * size_t(size.height);
	std::vector<pixel> image_data;
	image_data.resize(pixel_size);

	vkc::vkc gpu;

	fea::bench::suite suite;
	suite.title("GPU Tasks");
	suite.benchmark("Mandelbrot generator (pull only)", [&]() {
		vkc::task t{ gpu, shader_path.c_str() };

		t.push_constant("p_constants", size);
		t.reserve_buffer<pixel>("buf", image_data.size());
		t.record(size.width, size.height);
		t.submit();

		t.pull_buffer("buf", &image_data);
	});

	suite.print();


	// Save png.
	{
		std::vector<uint8_t> image;
		image.resize(image_data.size() * 4);

		// Cast data to 8bit color.
		for (size_t i = 0; i < image_data.size(); ++i) {
			image[i * 4] = uint8_t(image_data[i].r * 255.f);
			image[i * 4 + 1] = uint8_t(image_data[i].g * 255.f);
			image[i * 4 + 2] = uint8_t(image_data[i].b * 255.f);
			image[i * 4 + 3] = uint8_t(image_data[i].a * 255.f);
		}

		// Now we save the acquired color data to a .png.
		if (!stbi_write_png("mandelbrot.png", int(size.width), int(size.height),
					4, image.data(), int(size.width) * 4)) {
			fprintf(stderr, "Write png error : %s\n", stbi_failure_reason());
		}
	}
}
} // namespace

int main(int argc, char** argv) {
	argv0 = argv[0];

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
