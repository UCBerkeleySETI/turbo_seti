extern "C" __global__
void hitsearch_float64(const int n, const double* spectrum, const double threshold, const double drift_rate, double* maxsnr, double* maxdrift, unsigned int* tot_hits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = 0;
    for (int i = index; i < n; i += stride) {
        if (spectrum[i] > threshold) {
            count++;
            if (spectrum[i] > maxsnr[i]) {
                maxsnr[i] = spectrum[i];
                maxdrift[i] = drift_rate;
            }
        }
    }
    atomicAdd(&tot_hits[0], count);
}

extern "C" __global__
void hitsearch_float32(const int n, const float* spectrum, const double threshold, const double drift_rate, float* maxsnr, float* maxdrift, unsigned int* tot_hits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = 0;
    for (int i = index; i < n; i += stride) {
        if (spectrum[i] > threshold) {
            count++;
            if (spectrum[i] > maxsnr[i]) {
                maxsnr[i] = spectrum[i];
                maxdrift[i] = drift_rate;
            }
        }
    }
    atomicAdd(&tot_hits[0], count);
}
