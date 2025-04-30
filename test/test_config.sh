ASAN_OPTIONS=protect_shadow_gap=0 ./test_tensorrt
-O0 -g -fsanitize=address
export CUDA_MODULE_LOADING=LAZY