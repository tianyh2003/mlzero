rm -rf log
rm -rf build
rm -rf ckpt/weight_*
rm -rf ckpt/latest.ckpt
mkdir log
mkdir build
cd build
cmake ..
make -j 10
