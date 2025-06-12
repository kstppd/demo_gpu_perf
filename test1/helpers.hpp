#pragma once
#include "xxhash32.h"
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#ifdef USE_HIP
  #include <hip/hip_runtime.h>
  #define DEVICE_HOST __host__
  #define DEVICE_DEVICE __device__
  #define DEVICE_HOST_DEVICE __host__ __device__
  #define DEVICE_MALLOC hipMalloc
  #define DEVICE_FREE hipFree
  #define DEVICE_MEMCPY hipMemcpy
  #define DEVICE_SUCCESS hipSuccess
  #define DEVICE_MEMCPY_H2D hipMemcpyHostToDevice
  #define DEVICE_MEMCPY_D2H hipMemcpyDeviceToHost
  #define DEVICE_MEMCPY3D hipMemcpy3D
  #define DEVICE_MALLOC_3D_ARRAY hipMalloc3DArray
  #define DEVICE_CREATE_CHANNEL_DESC hipCreateChannelDesc
  #define DEVICE_MAKE_EXTENT make_hipExtent
  #define DEVICE_MAKE_PITCHED_PTR make_hipPitchedPtr
  #define DEVICE_CREATE_TEX_OBJ hipCreateTextureObject
  #define DEVICE_CREATE_SURF_OBJ hipCreateSurfaceObject
  #define DEVICE_RESOURCE_DESC hipResourceDesc
  #define DEVICE_TEXTURE_DESC hipTextureDesc
  #define DEVICE_RESOURCE_TYPE_ARRAY hipResourceTypeArray
  #define DEVICE_ADDR_MODE_WRAP hipAddressModeWrap
  #define DEVICE_FILTER_MODE_POINT hipFilterModePoint
  #define DEVICE_READ_MODE_ELEMENT hipReadModeElementType
  #define DEVICE_MEMSET hipMemset
  #define DEVICE_DEVICE_SYNC hipDeviceSynchronize
  #define DEVICE_DESTROY_TEX hipDestroyTextureObject
  #define DEVICE_FREE_ARRAY hipFreeArray
  #define DEVICE_ARRAY_T hipArray_t
  #define DEVICE_TEX_OBJ hipTextureObject_t
  #define DEVICE_MCPY_PARAMS hipMemcpy3DParms
  #define DEVICE_BND_MODE hipBoundaryModeTrap
  #define DEVICE_DESTROY_TEX_OBJ hipDestroyTextureObject


#else
  #include <cuda_device_runtime_api.h>
  #include <cuda_runtime_api.h>
  #include <driver_types.h>
  #define DEVICE_HOST __host__
  #define DEVICE_DEVICE __device__
  #define DEVICE_HOST_DEVICE __host__ __device__
  #define DEVICE_MALLOC cudaMalloc
  #define DEVICE_FREE cudaFree
  #define DEVICE_MEMCPY cudaMemcpy
  #define DEVICE_SUCCESS cudaSuccess
  #define DEVICE_MEMCPY_H2D cudaMemcpyHostToDevice
  #define DEVICE_MEMCPY_D2H cudaMemcpyDeviceToHost
  #define DEVICE_MEMCPY3D cudaMemcpy3D
  #define DEVICE_MALLOC_3D_ARRAY cudaMalloc3DArray
  #define DEVICE_CREATE_CHANNEL_DESC cudaCreateChannelDesc
  #define DEVICE_MAKE_EXTENT make_cudaExtent
  #define DEVICE_MAKE_PITCHED_PTR make_cudaPitchedPtr
  #define DEVICE_CREATE_TEX_OBJ cudaCreateTextureObject
  #define DEVICE_CREATE_SURF_OBJ cudaCreateSurfaceObject
  #define DEVICE_RESOURCE_DESC cudaResourceDesc
  #define DEVICE_TEXTURE_DESC cudaTextureDesc
  #define DEVICE_RESOURCE_TYPE_ARRAY cudaResourceTypeArray
  #define DEVICE_ADDR_MODE_WRAP cudaAddressModeWrap
  #define DEVICE_FILTER_MODE_POINT cudaFilterModePoint
  #define DEVICE_READ_MODE_ELEMENT cudaReadModeElementType
  #define DEVICE_MEMSET cudaMemset
  #define DEVICE_DEVICE_SYNC cudaDeviceSynchronize
  #define DEVICE_DESTROY_TEX cudaDestroyTextureObject
  #define DEVICE_FREE_ARRAY cudaFreeArray
  #define DEVICE_ARRAY_T cudaArray_t
  #define DEVICE_TEX_OBJ cudaTextureObject_t
  #define DEVICE_MCPY_PARAMS cudaMemcpy3DParms
  #define DEVICE_BND_MODE cudaBoundaryModeTrap
#endif

using type_t = float;
constexpr std::size_t N = 1 << 7;
constexpr type_t L = 1.0;
constexpr type_t DS = L / (N - 1);

DEVICE_HOST_DEVICE inline std::size_t
index(std::size_t i, std::size_t j, std::size_t k, std::size_t D) noexcept {
  return i + D * (j + D * k);
}

inline DEVICE_DEVICE void _3d_index_(std::size_t tid, std::size_t &i,
                                   std::size_t &j, std::size_t &k) noexcept {
  i = tid / (N * N);
  j = (tid % (N * N)) / N;
  k = tid % N;
}

inline constexpr std::array<std::size_t, 2>
launch_params(std::size_t arraySize, std::size_t blockSize) {
  std::size_t gridSize = (arraySize + blockSize - 1) / blockSize;
  return {gridSize, blockSize};
}

bool initialize_field(type_t *&ptr, type_t *&ptr2) {
  constexpr std::size_t bytes = N * N * N * sizeof(type_t);
  if (DEVICE_MALLOC(&ptr, bytes) != DEVICE_SUCCESS) return false;
  if (DEVICE_MALLOC(&ptr2, bytes) != DEVICE_SUCCESS) return false;
  if (!ptr || !ptr2) return false;

  std::vector<type_t> d(N * N * N, type_t{});
  constexpr type_t center = L / 2.0f;
  for (std::size_t k = 0; k < N; ++k)
    for (std::size_t j = 0; j < N; ++j)
      for (std::size_t i = 0; i < N; ++i) {
        type_t x = i * DS - center;
        type_t y = j * DS - center;
        type_t z = k * DS - center;
        type_t r2 = x * x + y * y + z * z;
        d.data()[index(i, j, k, N)] = std::exp(-r2);
      }

  if (DEVICE_MEMCPY(ptr, d.data(), bytes, DEVICE_MEMCPY_H2D) != DEVICE_SUCCESS)
    return false;

  return true;
}

bool destroy_field(type_t *&ptr) {
  if (ptr) {
    if (DEVICE_FREE(ptr) != DEVICE_SUCCESS) return false;
    ptr = nullptr;
    return true;
  }
  return false;
}

bool dump(const char *fname, type_t *ptr) {
  #ifndef DRY
  std::vector<type_t> data(N * N * N);
  DEVICE_MEMCPY(data.data(), ptr, N * N * N * sizeof(type_t), DEVICE_MEMCPY_D2H);
  std::ofstream file(fname, std::ios::binary);
  if (!file) return false;
  file.write(reinterpret_cast<const char *>(data.data()),
             data.size() * sizeof(type_t));
  file.close();
  #else
  (void)fname;
  (void)ptr;
  #endif

  return true;
}

uint32_t hash_array(type_t *ptr, std::size_t len) {
  std::vector<type_t> data(N * N * N);
  DEVICE_MEMCPY(data.data(), ptr, N * N * N * sizeof(type_t), DEVICE_MEMCPY_D2H);
  return XXHash32::hash(data.data(), len * sizeof(type_t), 0);
}

void get_cuda_array_3d(type_t *data_in, DEVICE_TEX_OBJ &texObj,
                       DEVICE_ARRAY_T array_in, bool write = false) {
  (void)write;
  auto desc = DEVICE_CREATE_CHANNEL_DESC<type_t>();
  auto ext = DEVICE_MAKE_EXTENT(N, N, N);
  auto err = DEVICE_MALLOC_3D_ARRAY(&array_in, &desc, ext, 0);
  if (err != DEVICE_SUCCESS) {
    fprintf(stderr, "ERROR: Failed to allocate 3D DEVICE Array with error: %d\n", err);
  }

  DEVICE_MCPY_PARAMS copyParams = {};
  copyParams.srcPtr = DEVICE_MAKE_PITCHED_PTR(data_in, N * sizeof(type_t), N, N);
  copyParams.dstArray = array_in;
  copyParams.extent = ext;
  copyParams.kind = DEVICE_MEMCPY_H2D;
  DEVICE_MEMCPY3D(&copyParams);

  DEVICE_RESOURCE_DESC resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = DEVICE_RESOURCE_TYPE_ARRAY;
  resDesc.res.array.array = array_in;

  DEVICE_TEXTURE_DESC texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = DEVICE_ADDR_MODE_WRAP;
  texDesc.addressMode[1] = DEVICE_ADDR_MODE_WRAP;
  texDesc.addressMode[2] = DEVICE_ADDR_MODE_WRAP;
  texDesc.filterMode = DEVICE_FILTER_MODE_POINT;
  texDesc.readMode = DEVICE_READ_MODE_ELEMENT;
  texDesc.normalizedCoords = 0;

  DEVICE_CREATE_TEX_OBJ(&texObj, &resDesc, &texDesc, NULL);
}
