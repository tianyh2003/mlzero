# MLZero
This is an implementation of alphazero for playing 8x8 gomoku. We can get a great model in several hours.
The original repository exists at https://gitee.com/tianyh2003/mlzero

# Requirements
- gcc/g++ 11.4.0
- cmake 3.25.2
- TensorRT 10.9.0
- libtorch 2.5.1

# Geting Started
- build the project
```bash
mkdir build
cd build
cmake ..
make
```
- train the model
```bash
./mlzero
```