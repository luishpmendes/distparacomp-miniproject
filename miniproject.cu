#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
//#include "cutil_inline.h"

#define GRIDSIZE 256
#define BLOCKSIZE 64
#define N 16
#define L -128.0
#define U 128.0
#define T 4096
//#define B 16.0
//#define ALPHA 0.5
#define RHO 64

using namespace std;

float randomUniform (float a, float b) {
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

int randInt (int a, int b) {
    int result;
    if (a <= b) {
        result = a + rand() % (b - a);
    } else {
        result = b + rand() % (a - b);
    }
    return result;
}
/*
float objectiveFunction (float * x) {
    float result = 0.0;
    for (int i = 0; i < N; i++) {
        result += x[i] * x[i];
    }
    return result;
}
*/
//http://www.sfu.ca/~ssurjano/rastr.html
//http://www.sfu.ca/~ssurjano/griewank.html
float objectiveFunction (float * x) {
    float result = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            result += x[j] * x[j];
        }
    }
    return result;
}

void initialSolution (float * x) {
    for (int i = 0; i < N; i++) {
        x[i] = randomUniform(L, U);
    }
}
/*
void blxAlphaCrossover (float * h, float * x0, float * x1, float alpha) {
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
        h[i] = randomUniform(hMin - I * alpha, hMax + I * alpha);
    }
}
*/

void crossover (float * h, float * x0, float * x1) {
    for (int i = 0; i < N; i++) {
        h[i] = randomUniform(x0[i], x1[i]);
    }
}

/*
float delta (int t, float y, float b) {
    float result = randomUniform(0, 1);
    result = pow (result, static_cast <float> (1.0 - static_cast <float> (t/T)));
    result = static_cast <float> (1 - result);
    result = pow (result, b);
    result = static_cast <float> (y * result);
    return result;
}

void nonUniformMutation (float * y, float * x, int t, float b) {
    int k = randInt(0, N);
    for (int i = 0; i < N; i++) {
        if (i == k) {
            if (randomUniform(0, 1) >= 0.5) {
                y[i] = static_cast <float> (x[i] + delta(t, static_cast <float> (U - x[i]), b));
            } else {
                y[i] = static_cast <float> (x[i] - delta(t, static_cast <float> (x[i] - L), b));
            }
        } else {
            y[i] = x[i];
        }
    }
}
*/

void mutation (float * y, float * x) {
    int k = randInt(0, N);
    for (int i = 0; i < N; i++) {
        if (i == k) {
            y[i] = randomUniform(L, U);
        } else {
            y[i] = x[i];
        }
    }
}

void host_findOptimum (float * solution) {
    float x0[N];
    float x1[N];
    initialSolution(x0);
    initialSolution(x1);
    for (int t = 0; t < T; t++) {
        float h[N];
        crossover(h, x0, x1);
        float y[N];
        mutation(y, h);
        if (objectiveFunction(x0) > objectiveFunction(x1)) {
            if (objectiveFunction(x0) > objectiveFunction(y)) {
                for (int i = 0; i < N; i++) {
                    x0[i] = y[i];
                }
            }
        } else {
            if (objectiveFunction(x1) > objectiveFunction(y)) {
                for (int i = 0; i < N; i++) {
                    x1[i] = y[i];
                }
            }
        }
    }
    if (objectiveFunction(x0) < objectiveFunction(x1)) {
        for (int i = 0; i < N; i++) {
            solution[i] = x0[i];
        }
    } else {
        for (int i = 0; i < N; i++) {
            solution[i] = x1[i];
        }
    }
}

int main (int argc, char** argv) {
    srand (time(NULL));
//    int devID;
//    cudaDeviceProp props;

    // get number of SMs on this GPU
//    cutilSafeCall(cudaGetDevice(&devID));
//    cutilSafeCall(cudaGetDeviceProperties(&props, devID));

    // allocate host memory
    unsigned int solutionMemSize = N * sizeof(float);

    float * hostSolution = (float *) malloc(solutionMemSize);

//    float * hostDeviceSolution = (float *) malloc(solutionMemSize);

    printf("Solution size : %d\n", N);
    printf("Grid size     : %d\n", GRIDSIZE);
    printf("Block size    : %d\n", BLOCKSIZE);

    // allocate device memory
//    float * deviceSolution;
//    cutilSafeCall(cudaMalloc((void**) &deviceSolution, solutionMemSize));

    // set up kernel for execution
//    printf("Run %d Kernels.\n\n", ITERS);
//    unsigned int timer = 0;
//    cutilCheckError(cutCreateTimer(&timer));
//    cutilCheckError(cutStartTimer(timer));

//    device_findOptimum<<<GRIDSIZE, BLOCKSIZE>>>(deviceSolution);

    // check if kernel execution generated and error
//    cutilCheckMsg("Kernel execution failed");

    // wait for device to finish
//    cudaThreadSynchronize();

    // stop and destroy timer
//    cutilCheckError(cutStopTimer(timer));
//    double dSeconds = (cutGetTimerValue(timer)/ITERS)/(1000.0);

    //Log througput
//    printf("Time = %.4f s\n", dSeconds);
//    cutilCheckError(cutDeleteTimer(timer));

    // copy result from device to host
//    cutilSafeCall(cudaMemcpy(hostDeviceSolution, deviceSolution, solutionMemSize, cudaMemcpyDeviceToHost));

    host_findOptimum(hostSolution);

    printf("%f\n", objectiveFunction(hostSolution));

    // clean up memory
//    free(hostDeviceSolution);
    free(hostSolution);
//    cutilSafeCall(cudaFree(deviceSolution));

    // exit and clean up device status
//    cudaThreadExit();

    return 0;
}
