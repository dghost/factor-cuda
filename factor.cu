/*
 *  CUDA square matrix multiplier
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include <stdint.h>

#include <math.h>
#include <cuda.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

/*
 *  Return elapsed wall time since last call (seconds)
 */
static double t0=0;
double Elapsed(void)
{
#ifdef _WIN32
    //  Windows version of wall time
    LARGE_INTEGER tv,freq;
    QueryPerformanceCounter((LARGE_INTEGER*)&tv);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    double t = tv.QuadPart/(double)freq.QuadPart;
#else
    //  Unix/Linux/OSX version of wall time
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double t = tv.tv_sec+1e-6*tv.tv_usec;
#endif
    double s = t-t0;
    t0 = t;
    return s;
}

/*
 *  Print message to stderr and exit
 */
void Fatal(const char* format , ...)
{
    va_list args;
    va_start(args,format);
    vfprintf(stderr,format,args);
    va_end(args);
    exit(1);
}

/*
 *  Initialize fastest GPU device
 */
int InitGPU(int verbose)
{
    //  Get number of CUDA devices
    int num;
    if (cudaGetDeviceCount(&num)) Fatal("Cannot get number of CUDA devices\n");
    if (num<1) Fatal("No CUDA devices found\n");
    
    //  Get fastest device
    cudaDeviceProp prop;
    int   MaxDevice = -1;
    int   MaxGflops = -1;
    for (int dev=0;dev<num;dev++)
    {
        if (cudaGetDeviceProperties(&prop,dev)) Fatal("Error getting device %d properties\n",dev);
        int Gflops = prop.multiProcessorCount * prop.clockRate;
        if (verbose) printf("CUDA Device %d: %s Gflops %f Processors %d Threads/Block %d\n",dev,prop.name,1e-6*Gflops,prop.multiProcessorCount,prop.maxThreadsPerBlock);
        if(Gflops > MaxGflops)
        {
            MaxGflops = Gflops;
            MaxDevice = dev;
        }
    }
    
    //  Print and set device
    if (cudaGetDeviceProperties(&prop,MaxDevice)) Fatal("Error getting device %d properties\n",MaxDevice);
    printf("Fastest CUDA Device %d: %s\n",MaxDevice,prop.name);
    cudaSetDevice(MaxDevice);
    
    //  Return max thread count
    return prop.maxThreadsPerBlock;
}

// host version of f(x)
__host__ __forceinline__ int64_t fx(int64_t x, int64_t a, int64_t c) {
    return ( a * x * x + c);
}

// host version of a binary gcd algorithm
__host__ int64_t gcd(int64_t u, int64_t v)
{
    int shift;
    
    /* GCD(0,v) == v; GCD(u,0) == u, GCD(0,0) == 0 */
    if (u == 0) return v;
    if (v == 0) return u;
    
    /* Let shift := lg K, where K is the greatest power of 2
     dividing both u and v. */
    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
        u >>= 1;
        v >>= 1;
    }
    
    while ((u & 1) == 0)
        u >>= 1;
    
    /* From here on, u is always odd. */
    do {
        /* remove all factors of 2 in v -- they are not common */
        /*   note: v is not zero, so while will terminate */
        while ((v & 1) == 0)  /* Loop X */
            v >>= 1;
        
        /* Now u and v are both odd. Swap if necessary so u <= v,
         then set v = v - u (which is even). For bignums, the
         swapping is just pointer movement, and the subtraction
         can be done in-place. */
        if (u > v) {
            int64_t t = v; v = u; u = t;}  // Swap u and v.
        v = v - u;                       // Here v >= u.
    } while (v != 0);
    
    /* restore common factors of 2 */
    return u << shift;
}

// host version of the Pollard's Rho algorithm
__host__ int64_t pollardHost(int64_t num)
{
    
    
    int64_t max = sqrt(num);
    
    // catch easy cases
    if ( num % 2 == 0)
    {
        //       cout << "Found 2" << endl;
        return 2;
        
    } else if ( num % 3 == 0)
    {
        return 3;
    } else if ( max * max == num)
    {
        return max;
    }
    
    int64_t result = 0;
    bool quit = false;
    
    // if OpenMP is enabled, automatically run this block in parallel
    // using variable 'quit' to synchronize
#pragma omp parallel
    {
        //int64_t x = rand() % max + 1;
        int64_t x = 0;
        int64_t a = rand() % (max-1) + 1;
        int64_t c = rand() % (max-1) + 1;
        int64_t y, d, z;
        
        y = x;
        d = 1;
        
        do
        {
            x = fx(x,a,c) % num;
            y = fx(fx(y,a,c),a,c) % num;
            z = std::abs(x-y);
            d = gcd(z,num);
        } while (d == 1 && !quit );
        
        
        if (d != 1 && d != num )
        {
            quit = true;
            result = d;
        }
    }
    
    return result;
}

// device version of f(x)
__device__ __forceinline__ int64_t fx_d(int64_t x, int64_t a, int64_t c) {
    return ( a * x * x + c);
}

// device version of binary gcd
__device__ int64_t gcd_d(int64_t u, int64_t v)
{
    int shift;
    
    /* GCD(0,v) == v; GCD(u,0) == u, GCD(0,0) == 0 */
    if (u == 0) return v;
    if (v == 0) return u;
    
    /* Let shift := lg K, where K is the greatest power of 2
     dividing both u and v. */
    for (shift = 0; ((u | v) & 1) == 0; ++shift) {
        u >>= 1;
        v >>= 1;
    }
    
    while ((u & 1) == 0)
        u >>= 1;
    
    /* From here on, u is always odd. */
    do {
        /* remove all factors of 2 in v -- they are not common */
        /*   note: v is not zero, so while will terminate */
        while ((v & 1) == 0)  /* Loop X */
            v >>= 1;
        
        /* Now u and v are both odd. Swap if necessary so u <= v,
         then set v = v - u (which is even). For bignums, the
         swapping is just pointer movement, and the subtraction
         can be done in-place. */
        if (u > v) {
            int64_t t = v; v = u; u = t;}  // Swap u and v.
        v = v - u;                       // Here v >= u.
    } while (v != 0);
    
    /* restore common factors of 2 */
    return u << shift;
}


// CUDA kernel for Pollard's Rho
// Only execute a single pass
__global__ void pollardKernel(int64_t num,int64_t* xd, int64_t* result)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t x,y,a,c,d, z;
    d = 1;

    // copy state variables back into local memory
    x = xd[threadID * 4];
    y = xd[threadID * 4 + 1];
    a = xd[threadID * 4 + 2];
    c = xd[threadID * 4 + 3];
    
    // execute the pass
    x = fx_d(x,a,c) % num;
    y = fx_d(fx_d(y,a,c),a,c) % num;
    z = abs(x-y);
    d = gcd_d(z,num);
    
    // copy updated state back into global memory
    xd[threadID * 4] = x;
    xd[threadID * 4 + 1] = y;

    // test to see if it found a factor
    if (d != 1 && d != num )
    {
        // if found, copy it into global syncronization variable "found"
        *result = d;
    }
    
}

// wrapper that sets up and calls pollardKernel
int64_t pollardDevice(int64_t num, int Bw, int Bn)
{
    //  Calculate matrix dimensions
    int n = 4 * Bw*Bn;
    int N =  n*sizeof(int64_t);
    
    // local variables
    int64_t max = sqrt(num);
    int64_t result = 0;
    
    // catch easy cases
    if ( num % 2 == 0)
    {
        return 2;
        
    } else if ( num % 3 == 0)
    {
        return 3;
    } else if ( max * max == num)
    {
        return max;
    }

    // initialize the state array
    int64_t *x;
    x = (int64_t *) malloc(N);
    if (!x) Fatal("Could not allocate host memory\n");

    
    for (int i=0 ; i < n ; i += 4)
    {
    	//x[i] = rand() % max + 1;
    	//x[1 + 1] = x[i];
        
        // set x, y, a, and c for each thread
        x[i] = 0;
        x[i + 1] = 0;
        x[i + 2] = rand() % (max-1) + 1;
    	x[i + 3] = rand() % (max-1) + 1;
        
    }
    // Allocate device memory
    int64_t* result_d;
    if (cudaMalloc((void**)&result_d,sizeof(int64_t))) Fatal("Cannot allocate device memory result_d\n");
    if (cudaMemcpy(result_d,&result,sizeof(int64_t),cudaMemcpyHostToDevice)) Fatal("Cannot copy result from host to device\n");

    int64_t* Xd;
    if (cudaMalloc((void**)&Xd,N)) Fatal("Cannot allocate device memory Ad\n");
    // do an asychronous copy operation and let the CUDA runtime sort out the details
    if (cudaMemcpyAsync(Xd,x,N,cudaMemcpyHostToDevice)) Fatal("Cannot copy X from host to device\n");
    
    // run the kernel until it finds a result
    do {
        pollardKernel<<<Bn,Bw>>>(num,Xd,result_d);
        
    	if (cudaMemcpy(&result,result_d,sizeof(int64_t),cudaMemcpyDeviceToHost)) {
            std::string error = cudaGetErrorString(cudaGetLastError());
            Fatal("%s\n", error.c_str());
        }
    } while (result == 0);
    
    // if it failed, abort
    if (cudaGetLastError()) Fatal("pollardKernel failed\n");
    
    //  Free device memory
    cudaFree(Xd);
    cudaFree(result_d);
    return result;
}

/*
 *  main
 */
int main(int argc, char* argv[])
{
    
    //  Process options
    int opt;
    int verbose=0;
    while ((opt=getopt(argc,argv,"v"))!=-1)
    {
        if (opt=='v')
            verbose++;
        else
            Fatal("Usage: [-v] <block width> <number of blocks>\n");
    }
    argc -= optind;
    argv += optind;
    
    
    //  Get width and number of blocks
    if (argc < 3) Fatal("Usage: [-v] <block width> <number of blocks> <numbers to check>\n");
    int Bw = atoi(argv[0]);
    if (Bw<1) Fatal("Block width out of range %d\n",Bw);
    int Bn = atoi(argv[1]);
    if (Bn<1) Fatal("Number of blocks out of range %d\n",Bn);
    
    
    //  Total width is block times number of blocks
    int n = Bw*Bn;
    printf("Bw=%d Bn=%d n=%d\n",Bw,Bn,n);
    
    //  Initialize GPU
    int Mw = InitGPU(verbose);
    if (Mw<Bw) Fatal("Thread count %d exceeds threads per block of %d\n",Bw*Bw,Mw);
    
    srand(time(NULL));
    double host_total = 0;
    double device_total = 0;
    
    for (int i = 2; i < argc; i++)
    {
#ifdef __i386__
        int64_t num = atoll(argv[i]);
#elif __amd64__
        int64_t num = atol(argv[i]);
#endif
        // Find factor on host
        Elapsed();
        int64_t res = 0;
        while (res == 0)
            res = pollardHost(num);
#ifdef __i386__
        printf("Host found common factor %lld of %lld\n",res,num);
#elif __amd64__
        printf("Host found common factor %ld of %ld\n",res,num);
#endif
        double Th = Elapsed();
        
        // test if the host found a valid factor
        bool pass = (num % res == 0);

        //  Find factor on device
        Elapsed();
        res = pollardDevice(num,Bw,Bn);
#ifdef __i386__
        printf("Device found common factor %lld of %lld\n",res,num);
#elif __amd64__
        printf("Device found common factor %ld of %ld\n",res,num);
#endif
        double Td = Elapsed();
        
        // test if both the host and the device found a valid factor
        pass = pass && (num % res == 0);
        
        //  Print results
        host_total += Th;
        device_total += Td;
        printf("Host   Time = %6.3f s\n",Th);
        printf("Device Time = %6.3f s\n",Td);
        printf("Speedup     = %.1f\n",Th/Td);
        printf("Result      = %s\n",(pass)?"PASS":"FAIL"); 
    }
    
    // print overall statistics
    printf("\nOverall results:\n");
    printf("Host   Time = %6.3f s\n",host_total);
    printf("Device Time = %6.3f s\n",device_total);
    printf("Speedup     = %.1f\n",host_total/device_total);
    printf("\n");

    //  Done
    return 0;
}
