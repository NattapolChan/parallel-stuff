#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define N 256
#define maxSize 128

__device__ int lock =0;

__global__ void inc(int* count){
    atomicAdd(count, 1);
}

__global__ void initCurandStates(curandStatePhilox4_32_10_t* states, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 65536) return;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void rand_append(int* d_queue, int* d_tail, curandStatePhilox4_32_10_t* rand_states) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    int insert_batch = (int)curand(&rand_states[i % 65536]);
    insert_batch = insert_batch % N;
    insert_batch = abs(insert_batch);
    // printf("%d \n ", insert_batch);
    int base = insert_batch * maxSize;
    int insert_idx = atomicAdd(&d_tail[insert_batch], 1);
    d_queue[base + insert_idx] = insert_idx;
}

int main()
{
    int* d_count, *h_count;
    int* h_queue = (int*)malloc(N * maxSize * sizeof(int));
    int* h_tail = (int*)malloc(N * sizeof(int));
    int* d_tail, *d_queue;
    curandStatePhilox4_32_10_t* curand_states;
    cudaMalloc((void **)&curand_states, 65536 * sizeof(curandStatePhilox4_32_10_t));
    initCurandStates<<<256, 256>>>(curand_states, 1234ULL);

    cudaMalloc((void **)&d_queue, N * maxSize * sizeof(int));
    cudaMalloc((void **)&d_tail, N * sizeof(int));
    h_count = (int*)malloc(sizeof(int));
    rand_append<<<8192,256>>>(d_queue, d_tail, curand_states);
    cudaDeviceSynchronize();
    cudaMemcpy(h_tail, d_tail, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_queue, d_queue, N * maxSize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i=0;i<N;i++) {
        for (int j=0;j<maxSize;j++) {
            printf("%d ", h_queue[i*maxSize + j]);
        }
        printf("\n");
    }
    printf("======================================\n");
    int sum_tail = 0;
    for(int i=0;i<N;i++) {
        sum_tail += h_tail[i];
    }
    printf("checking tail : %d\n", sum_tail);
    return 0;
}
