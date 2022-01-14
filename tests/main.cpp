#include <bench_util/bench_util.h>
#include <gtest/gtest.h>
#include <stb_image.h>
#include <vulkan_compute/vulkan_compute.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <intrin.h>

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
	size_block size;
	size_t pixel_size = size.width * size_t(size.height);
	std::vector<pixel> image_data;
	image_data.resize(pixel_size);

	vkc::vkc gpu{ argv0 };

	bench::suite suite;
	suite.title("Simulate GPU Modifier");
	suite.benchmark("Mandelbrot generator (no push, just pull)", [&]() {
		vkc::task t{ gpu, "data/shaders/mandelbrot.comp.spv" };

		t.update_push_constant("p_constants", size);
		t.reserve_buffer<pixel>(gpu, "buf", pixel_size);
		t.record(size.width, size.height);
		t.submit(gpu);

		t.pull_buffer(gpu, "buf", &image_data);
	});

	suite.print();
	suite.clear();


	// Get the color data from the buffer, and cast it to bytes.
	// We save the data to a vector.
	std::vector<uint8_t> image;
	image.resize(image_data.size() * 4);

	// simd just for fun :)
	alignas(16) std::array<float, 4> pix;
	const __m128 mul_v = _mm_set1_ps(255.f);

	for (size_t i = 0; i < image_data.size(); ++i) {
		__m128 v0 = _mm_loadu_ps((float*)&image_data[i]);
		__m128 answer = _mm_mul_ps(v0, mul_v);
		_mm_store_ps(pix.data(), answer);

		image[i * 4] = uint8_t(pix[0]);
		image[i * 4 + 1] = uint8_t(pix[1]);
		image[i * 4 + 2] = uint8_t(pix[2]);
		image[i * 4 + 3] = uint8_t(pix[3]);
	}

	// Now we save the acquired color data to a .png.
	int error = stbi_write_png("mandelbrot.png", int(size.width),
			int(size.height), 4, image.data(), int(size.width) * 4);

	if (error == 0) {
		fprintf(stderr, "write_png error %d : %s\n", error,
				stbi_failure_reason());
	}
}
} // namespace

int main(int argc, char** argv) {
	argv0 = argv[0];

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
