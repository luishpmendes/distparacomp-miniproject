#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include "cutil_inline.h"
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define GRIDSIZE 64
#define BLOCKSIZE 8
#define N 16
#define L -128.0
#define U 128.0
#define T 4096
#define TAU 64

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
/*
void host_blxAlphaCrossover (float * h, float * x0, float * x1, float alpha) {
    for (int i = 0; i < N; i++) {
        float hMin, hMax, I;
        if (x0[i] < x1[i]) {
            hMin = x0[i];
            hMax = x1[i];
        } else {
            hMin = x1[i];
            hMax = x0[i];
        }
        I = hMax - hMin;
        h[i] = host_randomUniform(hMin - I * alpha, hMax + I * alpha);
    }
}
*/

void host_crossover (float * h, float * x0, float * x1) {
    for (int i = 0; i < N; i++) {
        h[i] = host_randomUniform(x0[i], x1[i]);
    }
}

/*
float host_delta (int t, float y, float b) {
    float result = host_randomUniform(0, 1);
    result = pow (result, static_cast <float> (1.0 - static_cast <float> (t/T)));
    result = static_cast <float> (1 - result);
    result = pow (result, b);
    result = static_cast <float> (y * result);
    return result;
}

void host_nonUniformMutation (float * y, float * x, int t, float b) {
    int k = host_randInt(0, N);
    for (int i = 0; i < N; i++) {
        if (i == k) {
            if (host_randomUniform(0, 1) >= 0.5) {
                y[i] = static_cast <float> (x[i] + host_delta(t, static_cast <float> (U - x[i]), b));
            } else {
                y[i] = static_cast <float> (x[i] - host_delta(t, static_cast <float> (x[i] - L), b));
            }
        } else {
            y[i] = x[i];
        }
    }
}
*/

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

__device__ float device_randomUniform (curandState state, float a, float b) {
    //float result = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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

__device__ int device_randInt (curandState state, int a, int b) {
    int result;
    if (a <= b) {
        //result = a + rand() % (b - a);
        result = a + curand(state) % (b - a);
    } else {
        //result = b + rand() % (a - b);
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

__device__ void device_initialSolution (float * x, curandState state) {
    for (int i = 0; i < N; i++) {
        x[i] = device_randomUniform(curandState state, L, U);
    }
}
/*
__device__ void device_blxAlphaCrossover (float * h, float * x0, float * x1, float alpha) {
    for (int i = 0; i < N; i++) {
        float hMin, hMax, I;
        if (x0[i] < x1[i]) {
            hMin = x0[i];
            hMax = x1[i];
        } else {
            hMin = x1[i];
            hMax = x0[i];
        }
        I = hMax - hMin;
        h[i] = device_randomUniform(hMin - I * alpha, hMax + I * alpha);
    }
}
*/

__device__ void device_crossover (float * h, curandState state, float * x0, float * x1) {
    for (int i = 0; i < N; i++) {
        h[i] = device_randomUniform(state, x0[i], x1[i]);
    }
}

/*
__device__ float device_delta (int t, float y, float b) {
    float result = device_randomUniform(0, 1);
    result = pow (result, static_cast <float> (1.0 - static_cast <float> (t/T)));
    result = static_cast <float> (1 - result);
    result = pow (result, b);
    result = static_cast <float> (y * result);
    return result;
}

__device__ void device_nonUniformMutation (float * y, float * x, int t, float b) {
    int k = device_randInt(0, N);
    for (int i = 0; i < N; i++) {
        if (i == k) {
            if (device_randomUniform(0, 1) >= 0.5) {
                y[i] = static_cast <float> (x[i] + device_delta(t, static_cast <float> (U - x[i]), b));
            } else {
                y[i] = static_cast <float> (x[i] - device_delta(t, static_cast <float> (x[i] - L), b));
            }
        } else {
            y[i] = x[i];
        }
    }
}
*/

__device__ void device_mutation (float * y, curandState state, float * x) {
    int k = device_randInt(0, N);
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

    curandState state;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state);

    float x0[N];
    float x1[N];
    device_initialSolution(x0, state);
    device_initialSolution(x1, state);
    for (int t = 0; t < T; t++) {
        float h[N];
        device_crossover(h, state, x0, x1);
        float y[N];
        device_mutation(y, state, h);
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
        /*
            if t mod tau == 0
                write on shared mem // send the best individual -- the one with smallest objectiveFunction value
                sync threads
                read from share mem // we replace our worst individual only if the one we recieve is better
        */
    }
    if (device_objectiveFunction(x0) < device_objectiveFunction(x1)) {
        for (int i = 0; i < N; i++) {
            solution[i] = x0[i];
        }
    } else {
        for (int i = 0; i < N; i++) {
            solution[i] = x1[i];
        }
    }
    /*
        search all threads to find the best solution

        each thread put its better indvidual on shared memory
        sync threads
        thread zero search the shared memory to find the best solution 
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

    float * hostSolution = (float *) malloc(solutionMemSize);

    float * hostDeviceSolution = (float *) malloc(solutionMemSize);

    printf("Solution size : %d\n", N);
    printf("Grid size     : %d\n", GRIDSIZE);
    printf("Block size    : %d\n", BLOCKSIZE);

    // allocate device memory
    float * deviceSolution;
    cutilSafeCall(cudaMalloc((void**) &deviceSolution, solutionMemSize));

    // set up kernel for execution
//    printf("Run %d Kernels.\n\n", ITERS);
    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    device_findOptimum<<<GRIDSIZE, BLOCKSIZE>>>(deviceSolution, time(NULL));

    // loop {
    // start here
//    device_findOptimum<<<GRIDSIZE, BLOCKSIZE>>>(deviceSolution);
    // stop here
    // copy the solution found
    // copy the time delay
    // }
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // wait for device to finish
    cudaThreadSynchronize();

    // stop and destroy timer
    cutilCheckError(cutStopTimer(timer));
    double dSeconds = (cutGetTimerValue(timer)/1)/(1000.0);

    //Log througput
    printf("Time = %.4f s\n", dSeconds);
    cutilCheckError(cutDeleteTimer(timer));

    // copy result from device to host
    cutilSafeCall(cudaMemcpy(hostDeviceSolution, deviceSolution, solutionMemSize, cudaMemcpyDeviceToHost));

    host_findOptimum(hostSolution);

    printf("host: %f\n", host_objectiveFunction(hostSolution));
    printf("device: %f\n", host_objectiveFunction(hostDeviceSolution));

    // clean up memory
    free(hostDeviceSolution);
    free(hostSolution);
    cutilSafeCall(cudaFree(deviceSolution));

    // exit and clean up device status
    cudaThreadExit();

    return 0;
}
