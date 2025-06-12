#include "helpers.hpp"
#define BLOCK 8
#define LOOPS 8
#ifndef NOPROFILE
#ifdef __CUDACC__
#include <nvToolsExt.h>
#define PROFILE_START(msg) nvtxRangePushA((msg))
#define PROFILE_END() nvtxRangePop()
#else
#include <roctx.h>
#define PROFILE_START(msg) roctxRangePush((msg))
#define PROFILE_END() roctxRangePop()
#endif
#else
#define PROFILE_START(msg)
#define PROFILE_END()
#endif

__global__ void gradF(type_t *data_in, type_t *data_out, std::size_t len) {
  const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::size_t i, j, k;
  _3d_index_(tid, i, j, k);
  if (i == 0 || j == 0 || k == 0 || i == N - 1 || j == N - 1 || k == N - 1 ||
      tid >= len) {
    return;
  }

  constexpr auto denom = 2.0f * DS;
  type_t grad_x = std::abs(
      (data_in[index(i + 1, j, k, N)] - data_in[index(i - 1, j, k, N)]) /
      (denom));
  type_t grad_y = std::abs(
      (data_in[index(i, j + 1, k, N)] - data_in[index(i, j - 1, k, N)]) /
      (denom));
  type_t grad_z = std::abs(
      (data_in[index(i, j, k + 1, N)] - data_in[index(i, j, k - 1, N)]) /
      (denom));
  data_out[index(i, j, k, N)] = grad_x + grad_y + grad_z;
  return;
}

__global__ void gradF_tiled(type_t *data_in, type_t *data_out,
                            std::size_t len) {
  constexpr int TILE_SIZE = BLOCK;
  constexpr int SHARED_SIZE = TILE_SIZE + 2;
  __shared__ type_t tile[SHARED_SIZE * SHARED_SIZE * SHARED_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;

  const int i = blockIdx.x * TILE_SIZE + tx;
  const int j = blockIdx.y * TILE_SIZE + ty;
  const int k = blockIdx.z * TILE_SIZE + tz;

  const int sx = tx + 1;
  const int sy = ty + 1;
  const int sz = tz + 1;

  if (i < N && j < N && k < N) {
    tile[index(sx, sy, sz, SHARED_SIZE)] = data_in[index(i, j, k, N)];
  }

  if (tx == 0 && i > 0)
    tile[index(sx - 1, sy, sz, SHARED_SIZE)] = data_in[index(i - 1, j, k, N)];
  if (tx == TILE_SIZE - 1 && i < N - 1)
    tile[index(sx + 1, sy, sz, SHARED_SIZE)] = data_in[index(i + 1, j, k, N)];
  if (ty == 0 && j > 0)
    tile[index(sx, sy - 1, sz, SHARED_SIZE)] = data_in[index(i, j - 1, k, N)];
  if (ty == TILE_SIZE - 1 && j < N - 1)
    tile[index(sx, sy + 1, sz, SHARED_SIZE)] = data_in[index(i, j + 1, k, N)];
  if (tz == 0 && k > 0)
    tile[index(sx, sy, sz - 1, SHARED_SIZE)] = data_in[index(i, j, k - 1, N)];
  if (tz == TILE_SIZE - 1 && k < N - 1)
    tile[index(sx, sy, sz + 1, SHARED_SIZE)] = data_in[index(i, j, k + 1, N)];

  __syncthreads();

  if (i == 0 || i >= N - 1 || j == 0 || j >= N - 1 || k == 0 || k >= N - 1)
    return;

  constexpr auto denom = 2.0f * DS;
  const type_t dx = (tile[index(sx + 1, sy, sz, SHARED_SIZE)] -
                     tile[index(sx - 1, sy, sz, SHARED_SIZE)]) /
                    (denom);
  const type_t dy = (tile[index(sx, sy + 1, sz, SHARED_SIZE)] -
                     tile[index(sx, sy - 1, sz, SHARED_SIZE)]) /
                    (denom);
  const type_t dz = (tile[index(sx, sy, sz + 1, SHARED_SIZE)] -
                     tile[index(sx, sy, sz - 1, SHARED_SIZE)]) /
                    (denom);

  data_out[index(i, j, k, N)] = std::abs(dx) + std::abs(dy) + std::abs(dz);
}

__global__ void gradF_tex(DEVICE_TEX_OBJ data_in, type_t *data_out,
                          std::size_t len) {
  const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::size_t i, j, k;
  _3d_index_(tid, i, j, k);
  if (i == 0 || j == 0 || k == 0 || i == N - 1 || j == N - 1 || k == N - 1 ||
      tid >= len) {
    return;
  }

  constexpr auto denom = 2.0f * DS;
  type_t grad_x = std::abs((tex3D<type_t>(data_in, i + 1, j, k) -
                            tex3D<type_t>(data_in, i - 1, j, k)) /
                           (denom));
  type_t grad_y = std::abs((tex3D<type_t>(data_in, i, j + 1, k) -
                            tex3D<type_t>(data_in, i, j - 1, k)) /
                           (denom));
  type_t grad_z = std::abs((tex3D<type_t>(data_in, i, j, k + 1) -
                            tex3D<type_t>(data_in, i, j, k - 1)) /
                           (denom));
  data_out[index(i, j, k, N)] = grad_x + grad_y + grad_z;
  return;
}

__global__ void gradF_tiled_tex_in(DEVICE_TEX_OBJ data_in,
                                   type_t *data_out, std::size_t len) {
  constexpr int TILE_SIZE = BLOCK;
  constexpr int SHARED_SIZE = TILE_SIZE + 2;
  __shared__ type_t tile[SHARED_SIZE * SHARED_SIZE * SHARED_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;

  const int i = blockIdx.x * TILE_SIZE + tx;
  const int j = blockIdx.y * TILE_SIZE + ty;
  const int k = blockIdx.z * TILE_SIZE + tz;

  const int sx = tx + 1;
  const int sy = ty + 1;
  const int sz = tz + 1;

  if (i < N && j < N && k < N) {
    tile[index(sx, sy, sz, SHARED_SIZE)] = tex3D<type_t>(data_in, i, j, k);
  }

  if (tx == 0 && i > 0)
    tile[index(sx - 1, sy, sz, SHARED_SIZE)] =
        tex3D<type_t>(data_in, i - 1, j, k);
  if (tx == TILE_SIZE - 1 && i < N - 1)
    tile[index(sx + 1, sy, sz, SHARED_SIZE)] =
        tex3D<type_t>(data_in, i + 1, j, k);
  if (ty == 0 && j > 0)
    tile[index(sx, sy - 1, sz, SHARED_SIZE)] =
        tex3D<type_t>(data_in, i, j - 1, k);
  if (ty == TILE_SIZE - 1 && j < N - 1)
    tile[index(sx, sy + 1, sz, SHARED_SIZE)] =
        tex3D<type_t>(data_in, i, j + 1, k);
  if (tz == 0 && k > 0)
    tile[index(sx, sy, sz - 1, SHARED_SIZE)] =
        tex3D<type_t>(data_in, i, j, k - 1);
  if (tz == TILE_SIZE - 1 && k < N - 1)
    tile[index(sx, sy, sz + 1, SHARED_SIZE)] =
        tex3D<type_t>(data_in, i, j, k + 1);

  __syncthreads();

  if (i == 0 || i >= N - 1 || j == 0 || j >= N - 1 || k == 0 || k >= N - 1)
    return;

  constexpr auto denom = 2.0f * DS;
  const type_t dx = (tile[index(sx + 1, sy, sz, SHARED_SIZE)] -
                     tile[index(sx - 1, sy, sz, SHARED_SIZE)]) /
                    (denom);
  const type_t dy = (tile[index(sx, sy + 1, sz, SHARED_SIZE)] -
                     tile[index(sx, sy - 1, sz, SHARED_SIZE)]) /
                    (denom);
  const type_t dz = (tile[index(sx, sy, sz + 1, SHARED_SIZE)] -
                     tile[index(sx, sy, sz - 1, SHARED_SIZE)]) /
                    (denom);

  data_out[index(i, j, k, N)] = std::abs(dx) + std::abs(dy) + std::abs(dz);
}


int main(void) {

  PROFILE_START("Allocation + Initialization");
  type_t *data_in = nullptr;
  type_t *data_out = nullptr;
  if (!initialize_field(data_in, data_out)) {
    fprintf(stderr, "ERROR: Failed to allocate or initialize data!\n");
    return 1;
  }
  PROFILE_END();

  if (!dump("input.bin", data_in)) {
    fprintf(stderr, "ERROR: Could not dump input/output data to file!\n");
    return 1;
  }

  // ***************Launch Parameters*********************
  dim3 blockDim(BLOCK, BLOCK, BLOCK);
  dim3 gridDim((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK,
               (N + BLOCK - 1) / BLOCK);
  constexpr auto lp = launch_params(N * N * N, 256);
  /*******************************************************/
  
  PROFILE_START("Warp up");
  for (int c = 0; c < LOOPS; ++c) {
    gradF<<<lp[0], lp[1]>>>(data_in, data_out, N * N * N);
  }
  PROFILE_END();

  /*****************Run Naive Kernel*********************/
  PROFILE_START("Naive Kernel");
  for (int c = 0; c < LOOPS; ++c) {
    gradF<<<lp[0], lp[1]>>>(data_in, data_out, N * N * N);
  }
  DEVICE_DEVICE_SYNC();
  PROFILE_END();
  if (!dump("output_gradF.bin", data_out)) {
    fprintf(stderr, "ERROR: Could not dump input/output data to file!\n");
    return 1;
  }
  auto hash_baseline = hash_array(data_out, N * N * N);
  DEVICE_MEMSET(data_out, 0, sizeof(type_t) * N * N * N);
  /*******************************************************/

  /*****************Run Tiling Kernel*********************/
  PROFILE_START("Tiling Kernel");
  for (int c = 0; c < LOOPS; ++c) {
    gradF_tiled<<<gridDim, blockDim>>>(data_in, data_out, N * N * N);
  }
  DEVICE_DEVICE_SYNC();
  PROFILE_END();
  if (!dump("output_gradF_tiled.bin", data_out)) {
    fprintf(stderr, "ERROR: Could not dump input/output data to file!\n");
    return 1;
  }
  // Zero out grads
  auto hash_tiled = hash_array(data_out, N * N * N);
  DEVICE_MEMSET(data_out, 0, sizeof(type_t) * N * N * N);

  if (hash_baseline != hash_tiled) {
    fprintf(stderr, "ERROR: Array hashes do not match!\n");
    return 1;
  }
  /*******************************************************/

  /*****************Run Tex Kernel*********************/
  DEVICE_ARRAY_T array_in = nullptr;
  DEVICE_ARRAY_T array_out = nullptr;
  DEVICE_TEX_OBJ texObj_in = {0};
  get_cuda_array_3d(data_in, texObj_in, array_in);

  // Run Tex Kernel
  PROFILE_START("Texture Kernel");
  for (int c = 0; c < LOOPS; ++c) {
    gradF_tex<<<lp[0], lp[1]>>>(texObj_in, data_out, N * N * N);
  }
  DEVICE_DEVICE_SYNC();
  PROFILE_END();
  if (!dump("output_gradF_tex.bin", data_out)) {
    fprintf(stderr, "ERROR: Could not dump input/output data to file!\n");
    return 1;
  }
  // Zero out grads
  auto hash_tex = hash_array(data_out, N * N * N);
  DEVICE_MEMSET(data_out, 0, sizeof(type_t) * N * N * N);

  if (hash_baseline != hash_tex) {
    fprintf(stderr, "ERROR (1): Array hashes do not match!\n");
    return 1;
  }
  /*******************************************************/

  /*****************Run Tex Tiling Kernel*********************/
  PROFILE_START("Texture 2 Kernel");
  for (int c = 0; c < LOOPS; ++c) {
    gradF_tiled_tex_in<<<gridDim, blockDim>>>(texObj_in, data_out, N * N * N);
  }
  DEVICE_DEVICE_SYNC();
  PROFILE_END();
  if (!dump("output_gradF_tiled_tex_in.bin", data_out)) {
    fprintf(stderr, "ERROR: Could not dump input/output data to file!\n");
    return 1;
  }
  // Zero out grads
  auto hash_tiled_tex_in = hash_array(data_out, N * N * N);
  DEVICE_MEMSET(data_out, 0, sizeof(type_t) * N * N * N);

  if (hash_baseline != hash_tiled_tex_in) {
    fprintf(stderr, "ERROR: Array hashes do not match!\n");
    return 1;
  }
  /*******************************************************/

  // De Init
  PROFILE_START("DeInit");
  DEVICE_DESTROY_TEX(texObj_in);
  DEVICE_FREE_ARRAY(array_in);
  DEVICE_FREE_ARRAY(array_out);

  if (!destroy_field(data_in) || !destroy_field(data_out)) {
    fprintf(stderr, "ERROR: Failed to deallocate data!\n");
    return 1;
  }
  PROFILE_END();
  return 0;
}
