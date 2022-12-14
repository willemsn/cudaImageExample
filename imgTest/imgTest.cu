/****************************************************************************
 * Created by Pete Willemsen on 06/27/22
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of cudaImageExample
 *
 * cudaImageExample is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * cudaImageExample is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with cudaImageExample. If not, see
 * <https://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

#include "png++/png.hpp"

#include "ArgsImgTest.h"

struct PixelData 
{
    float r;
    float g;
    float b;
};

// kernel takes floats for better precision in colors
// a is input
// writes output to b
__global__ void darken(float *a, float *b, int N) 
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    if (tid < N)
    {
        b[tid] = a[tid] * 0.5;
    }
    else {
        b[tid] = 1.0;
    }
}

__global__ void lighten(float *a, float *b, int N) 
{
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    if (tid < N)
    {
        b[tid] = a[tid] * 1.5;
    }
    else {
        b[tid] = 0.5;
    }
}


void launchDarken(int numBlocks, int BLOCK_SIZE, float* input, float *output, int N, int dev)
{
  auto kSTime = std::chrono::steady_clock::now();  

  cudaSetDevice(dev);

  darken<<<numBlocks, BLOCK_SIZE>>>(input, output, N);
  std::cout << "Kernel Launch (darken): " << cudaGetErrorName(cudaGetLastError()) << std::endl;

  auto kETime = std::chrono::steady_clock::now();
  std::cout << "Kernel (darken) time: " << std::chrono::duration_cast<std::chrono::milliseconds>(kETime - kSTime).count() << " msec" << std::endl;
}

void launchLighten(int numBlocks, int BLOCK_SIZE, float* input, float *output, int N, int dev)
{
  auto kSTime = std::chrono::steady_clock::now();
  
  cudaSetDevice(dev);
  
  lighten<<<numBlocks, BLOCK_SIZE>>>(input, output, N);
  std::cout << "Kernel Launch (lighten): " << cudaGetErrorName(cudaGetLastError()) << std::endl;

  auto kETime = std::chrono::steady_clock::now();
  std::cout << "Kernel (lighten) time: " << std::chrono::duration_cast<std::chrono::milliseconds>(kETime - kSTime).count() << " msec" << std::endl;
}


int main(int argc, char *argv[])
{
    ArgsImgTest args;
    args.process(argc, argv);

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    // ERROR HANDLING
    if (error_id != cudaSuccess) {
        std::cerr << "ERROR!   cudaGetDeviceCount returned "
                  << static_cast<int>(error_id) << "\n\t-> "
                  << cudaGetErrorString(error_id) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<unsigned long long> devMemory( deviceCount );

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
        std::cerr << "There are no available device(s) that support CUDA\n";
        exit(EXIT_FAILURE);
    } else {
        std::cout << "Detected " << deviceCount << " CUDA Capable device(s)" << std::endl;
    }

    int dev, driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev) 
    {
        // sets the current GPU
        cudaSetDevice(dev);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        std::cout << "\tCUDA Driver Version / Runtime Version: "
                  << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / "
                  << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;

        std::cout << "\tCUDA Capability Major/Minor version number: "
                  << deviceProp.major << "." << deviceProp.minor << std::endl;

        devMemory[dev] = deviceProp.totalGlobalMem;
        
            char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg, sizeof(msg),
                  "\t\tTotal amount of global memory: %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                  (unsigned long long)deviceProp.totalGlobalMem);
#else
        snprintf(msg, sizeof(msg),
                 "\t\tTotal amount of global memory: %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long)deviceProp.totalGlobalMem);
#endif
        std::cout << msg;

        //    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
        //           deviceProp.multiProcessorCount,
        //           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
        //           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
        //           deviceProp.multiProcessorCount);

        std::cout << "\t\tGPU Max Clock rate:  "
                  << deviceProp.clockRate * 1e-3f << " MHz ("
                  << deviceProp.clockRate * 1e-6f << " GHz)" << std::endl;

        std::cout << "\t\tPCI: BusID=" << deviceProp.pciBusID << ", "
                  << "DeviceID=" << deviceProp.pciDeviceID << ", "
                  << "DomainID=" << deviceProp.pciDomainID << std::endl;
    }

    cudaSetDevice(0);

    bool writeToOneOutput = args.writeToOneOutput;
    std::cout << "Writing to one output = " << writeToOneOutput << std::endl;
    
    std::cout << "Reading image from file: " << args.inputFileName << std::endl;
    png::image< png::rgb_pixel > inputImage;

    inputImage.read( args.inputFileName );

    int imageWidth = inputImage.get_width();
    int imageHeight = inputImage.get_height();

    int numBytes = imageWidth * imageHeight * sizeof(PixelData);
    std::cout << "Number of bytes for file (" << numBytes << " bytes)" << std::endl;

    if (numBytes > devMemory[0]) {
        std::cout << "Warning: file size exceeds GPU memory limits." << std::endl;
    }

    // 
    // Memory allocation
    // 
    auto mSTime = std::chrono::steady_clock::now();  

    bool useManaged = true;
    
    float *hd_input0, *hd_input1;;
    float *hd_output0, *hd_output1;
    
    std::vector<float> host_input;
    std::vector<float> host_output0;
    std::vector<float> host_output1;

    if (!useManaged) {
      host_input.resize( imageWidth * imageHeight * 3 );
      host_output0.resize( imageWidth * imageHeight * 3 );
      host_output1.resize( imageWidth * imageHeight * 3 );      
    }

    auto cudaErrorVal = cudaGetLastError();
    if (useManaged) {
      cudaErrorVal = cudaMallocManaged(&hd_input0, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      
      cudaErrorVal = cudaMallocManaged(&hd_output0, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;

      cudaErrorVal = cudaMallocManaged(&hd_input1, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;

      cudaErrorVal = cudaMallocManaged(&hd_output1, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;

      // read in an image into data to be operated on by the kernel
      for (auto y=0; y<imageHeight; ++y) {
        for (auto x=0; x<imageWidth; ++x) {
                
	  png::rgb_pixel color;
	  color = inputImage[y][x];

	  auto idx = y * imageWidth * 3 + x * 3;

	  hd_input0[ idx ] = color.red/255.0;
	  hd_input0[ idx+1 ] = color.green/255.0;
	  hd_input0[ idx+2 ] = color.blue/255.0;
            
	  hd_input1[ idx ] = color.red/255.0;
	  hd_input1[ idx+1 ] = color.green/255.0;
	  hd_input1[ idx+2 ] = color.blue/255.0;
            
	  hd_output0[ idx ] = 0.0;
	  hd_output0[ idx+1 ] = 1.0;
	  hd_output0[ idx+2 ] = 0.0;
            
	  hd_output1[ idx ] = 1.0;	    
	  hd_output1[ idx+1 ] = 0.07;
	  hd_output1[ idx+2 ] = 0.57;
        }
      }

      // cudaMemAdvise(hd_input0, numBytes, cudaMemAdviseSetPreferredLocation, 0);
      // cudaMemAdvise(hd_output0, numBytes, cudaMemAdviseSetPreferredLocation, 0);
    }
    else {
      cudaErrorVal = cudaMalloc(&hd_input0, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      
      cudaErrorVal = cudaMalloc(&hd_output0, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;

      cudaErrorVal = cudaMalloc(&hd_input1, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;

      cudaErrorVal = cudaMalloc(&hd_output1, numBytes);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      
      for (auto y=0; y<imageHeight; ++y) {
        for (auto x=0; x<imageWidth; ++x) {
                
	  png::rgb_pixel color;
	  color = inputImage[y][x];

	  auto idx = y * imageWidth * 3 + x * 3;

	  host_input[ idx ] = color.red/255.0;
	  host_input[ idx+1 ] = color.green/255.0;
	  host_input[ idx+2 ] = color.blue/255.0;
            
	  host_output0[ idx ] = 0.0;
	  host_output0[ idx+1 ] = 1.0;
	  host_output0[ idx+2 ] = 0.0;
            
	  host_output1[ idx ] = 1.0;	    
	  host_output1[ idx+1 ] = 0.07;
	  host_output1[ idx+2 ] = 0.57;
        }
      }

      cudaErrorVal = cudaMemcpy(hd_input0, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      cudaErrorVal = cudaMemcpy(hd_output0, host_output0.data(), host_output0.size() * sizeof(float), cudaMemcpyHostToDevice);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      
      cudaErrorVal = cudaMemcpy(hd_input1, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      cudaErrorVal = cudaMemcpy(hd_output1, host_output1.data(), host_output1.size() * sizeof(float), cudaMemcpyHostToDevice);
      std::cout << "CUDA Error check: " << cudaGetErrorName(cudaErrorVal) << std::endl;
    }
    
    auto mETime = std::chrono::steady_clock::now();
    std::cout << "Mem time: " << std::chrono::duration_cast<std::chrono::milliseconds>(mETime - mSTime).count() << " msec" << std::endl;
    

    // Launch the kernel to do operations on the GPU
    int numElements = imageHeight*imageWidth*3;

    int BLOCK_SIZE = 512;
    int numBlocks = (numElements/2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "Num Blocks: " << numBlocks << ", BLOCK_SIZE = " << BLOCK_SIZE << std::endl;

    if (deviceCount == 2 && args.numGPUs == 2 && useManaged) {

      std::thread k0Thread(&launchDarken, numBlocks, BLOCK_SIZE, hd_input0, hd_output0, numElements/2, 0);
      std::thread k1Thread(&launchLighten, numBlocks, BLOCK_SIZE, hd_input0+numElements/2, hd_output0+numElements/2, numElements/2, 1 );

      // Sync
      auto kSTime = std::chrono::steady_clock::now();  
      cudaErrorVal = cudaDeviceSynchronize();
      std::cout << "Sync: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      if (deviceCount == 2 && args.numGPUs == 2 && useManaged) {
        cudaErrorVal = cudaDeviceSynchronize();
        std::cout << "Sync2: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      }
      auto kETime = std::chrono::steady_clock::now();
      std::cout << "Sync time: " << std::chrono::duration_cast<std::chrono::milliseconds>(kETime - kSTime).count() << " msec" << std::endl;

      k0Thread.join();
      k1Thread.join();
      
    }
    else {

      // =====================================================
      launchDarken( numBlocks, BLOCK_SIZE, hd_input0, hd_output0, numElements/2, 0 );

      // =====================================================       
      launchLighten( numBlocks, BLOCK_SIZE, hd_input0+numElements/2, hd_output0+numElements/2, numElements/2, 0 );

      // Sync
      auto kSTime = std::chrono::steady_clock::now();  
      cudaErrorVal = cudaDeviceSynchronize();
      std::cout << "Sync: " << cudaGetErrorName(cudaErrorVal) << std::endl;
      auto kETime = std::chrono::steady_clock::now();
      std::cout << "Sync time: " << std::chrono::duration_cast<std::chrono::milliseconds>(kETime - kSTime).count() << " msec" << std::endl;

    }
    
    // //////////////////////////////////////////////////////
    //
    // Write out all image data back to PNG files
    //
    // //////////////////////////////////////////////////////

    mSTime = std::chrono::steady_clock::now();  
    if (!useManaged) {
      cudaMemcpy(host_output0.data(), hd_output0, host_output0.size() * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(host_output1.data(), hd_output1, host_output1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    }
    mETime = std::chrono::steady_clock::now();
    std::cout << "Mem time: " << std::chrono::duration_cast<std::chrono::milliseconds>(mETime - mSTime).count() << " msec" << std::endl;



    // 
    // write out the image data from the kernel operations
    //
    mSTime = std::chrono::steady_clock::now();  
    
    png::image< png::rgb_pixel > outImage00( imageWidth, imageHeight );
    png::image< png::rgb_pixel > outImage01( imageWidth, imageHeight );
    for (auto y=0; y<imageHeight; ++y) {
        for (auto x=0; x<imageWidth; ++x) {
            
            auto idx = y * imageWidth * 3 + x * 3;
	    if (useManaged) {
	      outImage00[y][x] = png::rgb_pixel( hd_output0[idx]*255.0, hd_output0[idx+1]*255.0, hd_output0[idx+2]*255.0 );
	      outImage01[y][x] = png::rgb_pixel( hd_output1[idx]*255.0, hd_output1[idx+1]*255.0, hd_output1[idx+2]*255.0 );
	    }
	    else {
	      outImage00[y][x] = png::rgb_pixel( host_output0[idx]*255.0, host_output0[idx+1]*255.0, host_output0[idx+2]*255.0 );
	      outImage01[y][x] = png::rgb_pixel( host_output1[idx]*255.0, host_output1[idx+1]*255.0, host_output1[idx+2]*255.0 );
	    }
	}
    }
    outImage00.write( "gpuOutputImage_00.png" );
    outImage01.write( "gpuOutputImage_01.png" );

    mETime = std::chrono::steady_clock::now();
    std::cout << "Image Write time: " << std::chrono::duration_cast<std::chrono::milliseconds>(mETime - mSTime).count() << " msec" << std::endl;    
    
    cudaFree(hd_input0);
    cudaFree(hd_output0);
    cudaFree(hd_input1);
    cudaFree(hd_output1);
    
    exit(EXIT_SUCCESS);
}
