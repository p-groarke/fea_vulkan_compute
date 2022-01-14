# gpu_modifier

Prototype to benchmark and demonstrate a gpu compute modifier.

## Build
- You will need : An up-to-date cmake, MSVC and conan.
- Install Vulkan SDK : https://vulkan.lunarg.com/sdk/home#windows
- Add `C:\VulkanSDK\1.2.ver\Bin` to your Path environment variable.

### Windows
```
mkdir build && cd build
cmake .. -A x64 -DBUILD_TESTING=On && cmake --build .
bin\gpu_modifier_tests.exe
```
