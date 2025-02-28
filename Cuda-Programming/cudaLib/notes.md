
#### Streams Advanced


**Pinned Memory**

#### Memory that is locked in place and cannot be moved around in the OS. 

#### This is useful for when you want to move data around the GPU and do some computations on it.

#### If the OS moves the data around , the GPU will be looking for the data in the wrong place
#### and you will get segfault



````
```
// Allocate pinned memory
float* h_data;
cudaMallocHost((void**)&h_data, size);
```
````



**Events**

#### Measure critical execution time 
#### Computational costs are very minimal -- whole event will be syncronized within whole stream.



````
```
cudaEvent_t start, stop; // create these w/ memory adress of start and stop
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);    // plugin the streams
kernel<<<grid, block, 0, stream>>>(args); // event record , launch kerne;
cudaEventRecord(stop, stream); // another event record

cudaEventSynchronize(stop);  //
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);


```
````




**Callbacks**

#### When you want to log when something happens on a CPU we can use callbacks. 
#### More formally - " Completion of one operation on the GPU triggers the start of another operation on the CPU"

````
```
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\n");
    // Trigger next batch of work
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```
````




**CUDA libraries 📚**




**CUBLAS**

#### BLAS - short for B(asic) L(inear) A(lgebra) S(ubrrutines)

#### Also provides GEMM (general matrix multiplication) APIs with support for fusions that are highly optimized for NVIDIA GPUs

#### You'll be best off using the deep learning linear algebra operations in cuBLAS for matrix multiplication since it has wider coverage and is tuned for high throughput matmul.


#### The whole idea with CUBLAS is that you have this like black box obj that u call, and
### its opaque - encoded in binary and you cant see -API

#### Recommended to use perplexity, anthopic , good ol Google and just try and figure out
#### fastest inference for your GPU cluster irlt


#### Error checking 



````
```
    #define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

```
````

**CUBLAS-xt**

#### For massive matmuls :D


**CUTLASS**

#### Lets us fuse matrix operations together.





**CUDnn**

````
```
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


```

````


#### More indepth on this https://github.com/NVIDIA/CUDALibrarySamples



