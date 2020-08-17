#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>
#include <chrono>
#include <cooperative_groups.h>
using namespace std;

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define N_PROP 1100
#define THREADS_PER_BLOCK 256
#define N_BLOCKS 1    //Needed to be equal to 1 for the correct synchronization between threads

__device__ float dC = 0.0f;

__device__ int random_int(curandState &crstate, int min, int max){
    float myrandf = curand_uniform(&crstate);
    myrandf *= (max - min + 0.999999);
    myrandf += min;
    return (int)truncf(myrandf);
}

__global__ void trainBM(const float* weights, unsigned *net_state, const unsigned int net_size, 
                        const float temp_init, const float final_temp, const float cooling_rate, int seed) {

    // Handle to thread block group
    __shared__ float shared_weights_array[THREADS_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int nodes_per_thread = ceil((float)net_size/(float)THREADS_PER_BLOCK);
    int unit_index = 0;

    float temperature = temp_init;
    curandState crstate;
    curand_init(seed, 0, 0, &crstate);
    if( i<net_size ){
        net_state[i] = random_int(crstate, 0, 1);
    }

    do{
        __syncthreads();

        for(unsigned r=0; r<N_PROP; r++){

            unit_index = random_int(crstate, 0, net_size-1);
            if( i == 0 ){
                dC = 0.0f;
            }
            //if(i<3 && j<3)  printf("%d\n", unit_index);
            for(int k=0; k<nodes_per_thread; k++){
                int j = k*THREADS_PER_BLOCK + i;
                shared_weights_array[tid] = ( j<net_size && j!=unit_index) ? weights[unit_index*net_size+j]*net_state[j] : 0;

                __syncthreads();

                // do reduction in shared mem
                for (unsigned int s=blockDim.x/2; s>0; s>>=1)
                {
                    if (tid < s)
                    {
                        shared_weights_array[tid] += shared_weights_array[tid + s];
                    }

                    __syncthreads();
                }

                if (tid == 0){
                    dC += shared_weights_array[0];
                }
                __syncthreads();
            }  

            if (i == 0){

                float probability = 1 / (1 + expf(-dC / temperature));
                //printf("dC--------->%f\nPROBABILITY--------->%f\n", dC, probability);
                if( curand_uniform(&crstate)<probability ){
                    net_state[unit_index]=1;
                }else{
                    net_state[unit_index]=0;
                }

            }else{
                skipahead(1, &crstate);
            }

        }

        temperature*=cooling_rate;
        if(i==0)    printf("%f\n", temperature);   

    }while( temperature > final_temp );

}

/*Params: 1. Number of nodes for the network 
          2. Initial Temperature
          3. Final Temperature
          4. Cooling Rate                      
          5. Path to the file with the weights matrix
          6. Seed
*/
int main(int narg, char** arg){

    if(narg != 7){
        perror("Wrong number of arguments.");
        exit(1);
    }

    auto start = chrono::high_resolution_clock::now();

    const unsigned int NET_SIZE = atoi(arg[1]);
    const float INITIAL_TEMPERATURE = atof(arg[2]);
    const float FINAL_TEMPERATURE = atof(arg[3]);
    const float COOL_RATE = atof(arg[4]);
    const unsigned SEED = atoi(arg[5]);
    unsigned* net_states = new unsigned [NET_SIZE]; 
    srand(SEED);
    for(int i=0; i<NET_SIZE; i++){
        net_states[i] = rand() % 2;
    }
    float* weights = new float [NET_SIZE*NET_SIZE];
    ifstream matrix_file(arg[6], ifstream::in);
    for(int i=0; i<NET_SIZE; i++){
        for(int j=0; j<NET_SIZE; j++){
            string x;
            matrix_file>>x;
            weights[i*NET_SIZE+j] = strtof( x.c_str(), NULL );
        }
    }

    auto start_gpu = chrono::high_resolution_clock::now();

    //Moving data to GPU
    unsigned* g_net_states;
    CUDA_CALL( cudaMalloc((void **)&g_net_states, NET_SIZE*sizeof(unsigned)) );
    //CUDA_CALL( cudaMemcpy(g_net_states, net_states, NET_SIZE*sizeof(unsigned), cudaMemcpyHostToDevice) );
    float* g_weights;
    CUDA_CALL( cudaMalloc((float **)&g_weights, NET_SIZE*NET_SIZE*sizeof(float*)) );
    CUDA_CALL( cudaMemcpy(g_weights, weights, NET_SIZE*NET_SIZE*sizeof(float), cudaMemcpyHostToDevice) );


    //Running kernel
    unsigned threadsPerBlock = THREADS_PER_BLOCK;
    unsigned blocksPerGrid = N_BLOCKS;
    trainBM<<<blocksPerGrid, threadsPerBlock>>>(g_weights, g_net_states, NET_SIZE, 
                                                INITIAL_TEMPERATURE, FINAL_TEMPERATURE, COOL_RATE, SEED);
    /*std::string error = cudaGetErrorString(cudaPeekAtLastError());
    printf("%s\n", error.c_str());
    error = cudaGetErrorString(cudaThreadSynchronize());
    printf("%s\n", error.c_str());*/

    //Moving result to CPU
    CUDA_CALL( cudaMemcpy(net_states, g_net_states, NET_SIZE*sizeof(float), cudaMemcpyDeviceToHost) );

    //Freeing GPU memory
    CUDA_CALL( cudaFree(g_net_states) );
    CUDA_CALL( cudaFree(g_weights) );

    auto end_gpu = chrono::high_resolution_clock::now();

    float C = 0;
    for(int i=0; i<NET_SIZE; i++){
        for(int j=i+1; j<NET_SIZE; j++){
            if( net_states[i] & net_states[j] )   C += weights[i*NET_SIZE+j];
        }
    }

    delete [] net_states;
    delete [] weights;

    auto end = chrono::high_resolution_clock::now();

    cout<<"Final cost function: "<<C<<endl;
    cout<<"GPU time: "<<chrono::duration_cast<chrono::milliseconds>(end_gpu-start_gpu).count()/1000.0f<<" seconds."<<endl;
    cout<<"Total time: "<<chrono::duration_cast<chrono::milliseconds>(end-start).count()/1000.0f<<" seconds."<<endl;

    return 0;
}