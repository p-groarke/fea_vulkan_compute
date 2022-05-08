#include <fea/benchmark/benchmark.hpp>
#include <fea/utils/file.hpp>
#include <gtest/gtest.h>
#include <string>
#include <vkc/vulkan_compute.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

extern const char* argv0;

namespace {
struct pixel {
	float r = 0.f;
	float g = 0.f;
	float b = 0.f;
	float a = 1.f;
};

struct size_block {
	uint32_t width = 500;
	uint32_t height = 500;
};

TEST(vulkan_compute, basics) {
	std::filesystem::path exe_path = fea::executable_dir(argv0);
	std::wstring shader_path = exe_path / L"data/shaders/mandelbrot.comp.spv";

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
		t.submit(size.width, size.height, 1);

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

		std::filesystem::path out_filepath = exe_path / L"mandelbrot.png";

		// Now we save the acquired color data to a .png.
		if (!stbi_write_png(out_filepath.string().c_str(), int(size.width),
					int(size.height), 4, image.data(), int(size.width) * 4)) {
			fprintf(stderr, "Write png error : %s\n", stbi_failure_reason());
			ASSERT_TRUE(false);
		}
	}

	// Load expected png.
	{
		int width = 0;
		int height = 0;
		int chans = 0;

		const uint8_t* cmp_img = nullptr;
		const uint8_t* test_img = nullptr;

		{
			std::filesystem::path cmp_filepath
					= exe_path / L"data/images/mandelbrot.png";
			ASSERT_TRUE(std::filesystem::exists(cmp_filepath));

			cmp_img = stbi_load(cmp_filepath.string().c_str(), &width, &height,
					&chans, STBI_rgb_alpha);

			EXPECT_EQ(width, 500);
			EXPECT_EQ(height, 500);
			EXPECT_EQ(chans, 4);
		}

		{
			std::filesystem::path test_filepath = exe_path / L"mandelbrot.png";
			ASSERT_TRUE(std::filesystem::exists(test_filepath));

			int twidth = 0;
			int theight = 0;
			int tchans = 0;
			test_img = stbi_load(test_filepath.string().c_str(), &twidth,
					&theight, &tchans, STBI_rgb_alpha);

			EXPECT_EQ(width, twidth);
			EXPECT_EQ(height, theight);
			EXPECT_EQ(chans, tchans);
		}

		// TODO : fix in swiftshader
		// for (int h = 0; h < height; ++h) {
		//	for (int w = 0; w < width * chans; ++w) {
		//		int idx = h * width + w;
		//		EXPECT_EQ(cmp_img[idx], test_img[idx]);
		//	}
		//}
	}
}
} // namespace
