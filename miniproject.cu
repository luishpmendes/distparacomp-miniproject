#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include "cutil_inline.h"
#include <curand_kernel.h>

#define GRIDSIZE 1
#define BLOCKSIZE 16
#define N 16
#define L -128.0
#define U 128.0
#define T 4096
#define TAU 64
#define R 16

using namespace std;

float host_randomUniform (float a, float b) {
    float result = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    float min, max;
    if (a < b) {
        min = a;
        max = b;
    } else {
        min = b;
        max = a;
    }
    float diff = max - min;
    result *= diff;
    result += min;
    return result;
}

int host_randInt (int a, int b) {
    int result;
    if (a <= b) {
        result = a + rand() % (b - a);
    } else {
        result = b + rand() % (a - b);
    }
    return result;
}
/*
//Sphere function
float host_objectiveFunction (float * x) {
    float result = 0.0;
    for (int i = 0; i < N; i++) {
        result += x[i] * x[i];
    }
    return result;
}
*/
/*

float host_objectiveFunction (float * x) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            result += x[j] * x[j];
        }
    }
    return result;
}
*/
/*
//Griewank Function
float host_objectiveFunction (float * x) {
    float result = 1;
    float sum = 0;
    float prod = 1;
    for (int i = 0; i < N; i++) {
        sum += x[i] * x[i];
        prod *= cos(x[i]/sqrt(i+1));
    }
    sum /= 4000;
    result += sum;
    result -= prod;
    return result;
}
*/
//Rastrigin Function
float host_objectiveFunction (float * x) {
    float result = 10*N;
    for (int i = 0; i < N; i++) {
        result += x[i] * x[i];
        result -= 10*cos(2*M_PI*x[i]);
    }
    return result;
}

void host_initialSolution (float * x) {
    for (int i = 0; i < N; i++) {
        x[i] = host_randomUniform(L, U);
    }
}

void host_crossover (float * h, float * x0, float * x1) {
    for (int i = 0; i < N; i++) {
        h[i] = host_randomUniform(x0[i], x1[i]);
    }
}

void host_mutation (float * y, float * x) {
    int k = host_randInt(0, N);
    for (int i = 0; i < N; i++) {
        if (i == k) {
            y[i] = host_randomUniform(L, U);
        } else {
            y[i] = x[i];
        }
    }
}

void host_findOptimum (float * solution) {
    float x0[N];
    float x1[N];
    host_initialSolution(x0);
    host_initialSolution(x1);
    for (int t = 0; t < T; t++) {
        float h[N];
        host_crossover(h, x0, x1);
        float y[N];
        host_mutation(y, h);
        if (host_objectiveFunction(x0) > host_objectiveFunction(x1)) {
            if (host_objectiveFunction(x0) > host_objectiveFunction(y)) {
                for (int i = 0; i < N; i++) {
                    x0[i] = y[i];
                }
            }
        } else {
            if (host_objectiveFunction(x1) > host_objectiveFunction(y)) {
                for (int i = 0; i < N; i++) {
                    x1[i] = y[i];
                }
            }
        }
    }
    if (host_objectiveFunction(x0) < host_objectiveFunction(x1)) {
        for (int i = 0; i < N; i++) {
            solution[i] = x0[i];
        }
    } else {
        for (int i = 0; i < N; i++) {
            solution[i] = x1[i];
        }
    }
}

__device__ float device_randomUniform (curandState * state, float a, float b) {
    float result = curand_uniform(state);
    float min, max;
    if (a < b) {
        min = a;
        max = b;
    } else {
        min = b;
        max = a;
    }
    float diff = max - min;
    result *= diff;
    result += min;
    return result;
}

__device__ int device_randInt (curandState * state, int a, int b) {
    int result;
    if (a <= b) {
        result = a + curand(state) % (b - a);
    } else {
        result = b + curand(state) % (a - b);
    }
    return result;
}
/*
//Sphere function
__device__ float device_objectiveFunction (float * x) {
    float result = 0.0;
    for (int i = 0; i < N; i++) {
        result += x[i] * x[i];
    }
    return result;
}
*/
/*

__device__ float device_objectiveFunction (float * x) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            result += x[j] * x[j];
        }
    }
    return result;
}
*/
/*
//Griewank Function
__device__ float device_objectiveFunction (float * x) {
    float result = 1;
    float sum = 0;
    float prod = 1;
    for (int i = 0; i < N; i++) {
        sum += x[i] * x[i];
        prod *= cos(x[i]/sqrt(i+1));
    }
    sum /= 4000;
    result += sum;
    result -= prod;
    return result;
}
*/
//Rastrigin Function
__device__ float device_objectiveFunction (float * x) {
    float result = 10*N;
    for (int i = 0; i < N; i++) {
        result += x[i] * x[i];
        result -= 10*cos(2*M_PI*x[i]);
    }
    return result;
}

__device__ void device_initialSolution (curandState * state, float * x) {
    for (int i = 0; i < N; i++) {
        x[i] = device_randomUniform(state, L, U);
    }
}

__device__ void device_crossover (curandState * state, float * h, float * x0, float * x1) {
    for (int i = 0; i < N; i++) {
        h[i] = device_randomUniform(state, x0[i], x1[i]);
    }
}

__device__ void device_mutation (curandState * state, float * y, float * x) {
    int k = device_randInt(state, 0, N);
    for (int i = 0; i < N; i++) {
        if (i == k) {
            y[i] = device_randomUniform(state, L, U);
        } else {
            y[i] = x[i];
        }
    }
}

__global__ void device_findOptimum (float * solution, unsigned int seed) {
    // initialize shared mem

    __shared__ float sharedMem[GRIDSIZE*BLOCKSIZE][N];

    curandState state;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state);

    float x0[N];
    float x1[N];
    device_initialSolution(&state, x0);
    device_initialSolution(&state, x1);
    for (int t = 0; t < T; t++) {
        float h[N];
        device_crossover(&state, h, x0, x1);
        float y[N];
        device_mutation(&state, y, h);
        if (device_objectiveFunction(x0) > device_objectiveFunction(x1)) {
            if (device_objectiveFunction(x0) > device_objectiveFunction(y)) {
                for (int i = 0; i < N; i++) {
                    x0[i] = y[i];
                }
            }
        } else {
            if (device_objectiveFunction(x1) > device_objectiveFunction(y)) {
                for (int i = 0; i < N; i++) {
                    x1[i] = y[i];
                }
            }
        }
        if (t % TAU) {

            __syncthreads();

            // send best indvidual to the next thread through the shared memory

            if (device_objectiveFunction(x0) < device_objectiveFunction(x1)) { // x0 is the best
                for (int i = 0; i < N; i++) {
                    sharedMem[(id+1)%GRIDSIZE*BLOCKSIZE][i] = x0[i];
                }
            } else { // x1 is the best
                for (int i = 0; i < N; i++) {
                    sharedMem[(id+1)%GRIDSIZE*BLOCKSIZE][i] = x1[i];
                }
            }

            #if __CUDA_ARCH__>=200
                printf("thread %d is writing on memory\n", id);
            #endif

            __syncthreads();

            // get the individual from previous thread and replace worst individual if it is worse than the received one
            
            #if __CUDA_ARCH__>=200
                printf("thread %d is reading on memory\n", id);
            #endif

            #if __CUDA_ARCH__>=200
                printf("%d %f\n", id, device_objectiveFunction(sharedMem[id]));
            #endif

            if (device_objectiveFunction(x0) > device_objectiveFunction(x1)) { // x0 is the worst
                if (device_objectiveFunction(x0) > device_objectiveFunction(sharedMem[id])) { // x0 is worse than the received one
                    for (int i = 0; i < N; i++) {
                        x0[i] = sharedMem[id][i];
                    }
                }
            } else { // x1 is the worst
                if (device_objectiveFunction(x1) > device_objectiveFunction(sharedMem[id])) { // x1 is worse than the received one
                    for (int i = 0; i < N; i++) {
                        x1[i] = sharedMem[id][i];
                    }
                }
            }
            
        }
    }
    
    __syncthreads();
    
    // put the best individual on the shared memory

    if (device_objectiveFunction(x0) < device_objectiveFunction(x1)) { // x0 is the best
        for (int i = 0; i < N; i++) {
            sharedMem[id][i] = x0[i];
        }
    } else { // x1 is the best
        for (int i = 0; i < N; i++) {
            sharedMem[id][i] = x1[i];
        }
    }
    

    __syncthreads();

    // delegate the task of picking the best of the best to thread zero
    
    if (id == 0) {
        int idBest = 0;
        for (int i = 1; i < GRIDSIZE*BLOCKSIZE; i++) {
            if (device_objectiveFunction(sharedMem[i]) < device_objectiveFunction(sharedMem[idBest])) {
                idBest = i;
            }
        }
        for (int i = 0; i < N; i++) {
            solution[i] = sharedMem[idBest][i];
        }
    }
    
    // remove the following code
    /*
    if (device_objectiveFunction(x0) < device_objectiveFunction(x1)) {
        for (int i = 0; i < N; i++) {
            solution[i] = x0[i];
        }
    } else {
        for (int i = 0; i < N; i++) {
            solution[i] = x1[i];
        }
    }
    */
}

int main (int argc, char** argv) {
    srand (time(NULL));
    int devID;
    cudaDeviceProp props;

    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDevice(&devID));
    cutilSafeCall(cudaGetDeviceProperties(&props, devID));

    // allocate host memory
    unsigned int solutionMemSize = N * sizeof(float);

    float deviceSolutionValue[R];
    float deviceSolutionTime[R];

    float hostSolutionValue[R];
    float hostSolutionTime[R];

    printf("Solution size : %d\n", N);
    printf("Grid size     : %d\n", GRIDSIZE);
    printf("Block size    : %d\n", BLOCKSIZE);

    // allocate device memory
    float * deviceSolution;
    cutilSafeCall(cudaMalloc((void**) &deviceSolution, solutionMemSize));

    // set up kernel for execution
    printf("Run %d Kernels.\n\n", R);

    for (int r = 0; r < R; r++) {
        float hostDeviceSolution[N];
        unsigned int timer = 0;
        cutilCheckError(cutCreateTimer(&timer));
        cutilCheckError(cutStartTimer(timer));

        device_findOptimum<<<GRIDSIZE, BLOCKSIZE>>>(deviceSolution, time(NULL));

        // check if kernel execution generated and error
        cutilCheckMsg("Kernel execution failed");

        // wait for device to finish
        cudaThreadSynchronize();

        // stop and destroy timer
        cutilCheckError(cutStopTimer(timer));
        deviceSolutionTime[r] = cutGetTimerValue(timer)/(1000.0);
        cutilCheckError(cutDeleteTimer(timer));

        // copy result from device to host
        cutilSafeCall(cudaMemcpy(hostDeviceSolution, deviceSolution, solutionMemSize, cudaMemcpyDeviceToHost));
        deviceSolutionValue[r] = host_objectiveFunction(hostDeviceSolution);
    
        for (int i = 0; i < N; ++i){
            printf("solution[%d] = %f%s\n", i, hostDeviceSolution[i]);
        }
    }

    float deviceAverageSolutionValue = 0.0;
    float deviceAverageSolutionTime = 0.0;

    for (int r = 0; r < R; r++) {
        deviceAverageSolutionValue += deviceSolutionValue[r];
        deviceAverageSolutionTime += deviceSolutionTime[r];
    }

    deviceAverageSolutionValue /= R;
    deviceAverageSolutionTime /= R;

    for (int r = 0; r < R; r++) {
        clock_t begin, end;
        float hostSolution[N];
        
        begin = clock();
        host_findOptimum(hostSolution);
        end = clock();

        hostSolutionTime[r] = static_cast <float> (end - begin) / static_cast <float> (CLOCKS_PER_SEC);
        hostSolutionValue[r] = host_objectiveFunction(hostSolution);
    }

    float hostAverageSolutionValue = 0.0;
    float hostAverageSolutionTime = 0.0;

    for (int r = 0; r < R; r++) {
        hostAverageSolutionValue += hostSolutionValue[r];
        hostAverageSolutionTime += hostSolutionTime[r];
    }

    hostAverageSolutionValue /= R;
    hostAverageSolutionTime /= R;

    printf("Host objective function value : %f\n", hostAverageSolutionValue);
    printf("Host time: %f s\n", hostAverageSolutionTime);
    printf("Device objective function value: %f\n", deviceAverageSolutionValue);
    printf("Device time: %f s\n", deviceAverageSolutionTime);

    // clean up memory
    cutilSafeCall(cudaFree(deviceSolution));

    // exit and clean up device status
    cudaThreadExit();

    return 0;
}
