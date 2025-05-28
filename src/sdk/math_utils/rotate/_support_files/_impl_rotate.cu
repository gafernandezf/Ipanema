__global__ void transform_f32(float *in, float *out, float *T, int N)
  {
    int el = threadIdx.x + blockDim.x * blockIdx.x;
    int i0 = el*N;
    int i, j;
    for (i = 0; i <N; i+=1){
        out[i0 + i] = 0;
        for (j = 0; j<N; j+=1) {
           out[i0 + i]  +=  T[j + i*N]  *in[i0 +j];
        }
    }
    
  }