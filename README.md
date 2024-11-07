# CUDA-Matrix
This repository contains implementations of basic matrix operations using both CPU and GPU methods. It includes code for vector addition, matrix addition, and matrix multiplication, optimized for performance in CPU and GPU environments. 

# Getting Started
You need to install: 
- A functional GPU and a Linux Desktop
- CUDA Toolkit (required for GPU code execution)
- C++ compiler (for compiling CPU code)

# Build and run
## CPU
compile:
```
g++ -o matrix_multiply matrix_multiply.cpp -o cmat
```
run:
```
./cmat
```

## gpu
compile:
```
nvcc -arch=sm_86 matrix_multiply.cu -o gmat
```
run:
```
./gmat
```



