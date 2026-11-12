// Standalone CUDA benchmark for the hierarchical full-resolution motion search
// used by the hybrid x264 fork.  Not part of the encoder build.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef K
#define K 8
#endif

typedef struct { int16_t x, y; uint32_t sad; } Cand;

static void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

__device__ __forceinline__ uint32_t sad4x4(const uint8_t *a, int as, const uint8_t *b, int bs) {
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        #pragma unroll
        for (int i = 0; i < 4; i++)
            s += abs((int)a[j*as+i] - (int)b[j*bs+i]);
    }
    return s;
}

__device__ __forceinline__ uint32_t sad16x16(const uint8_t *a, int as, const uint8_t *b, int bs) {
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        #pragma unroll
        for (int i = 0; i < 16; i += 4) {
            uint32_t av = *reinterpret_cast<const uint32_t *>(a + j*as + i);
            uint32_t bv = *reinterpret_cast<const uint32_t *>(b + j*bs + i);
            s += __vabsdiffu4(av, bv);
            // __vabsdiffu4 is per-byte; the upper bytes are independent, but
            // summing the packed value only works if no byte overflows.  Sum
            // bytes explicitly instead.
        }
    }
    return s;
}

// Byte-accurate scalar version used by the reference benchmark.
__device__ __forceinline__ uint32_t sad16_scalar(const uint8_t *a, int as, const uint8_t *b, int bs) {
    uint32_t s = 0;
    #pragma unroll
    for (int j = 0; j < 16; j++)
        #pragma unroll
        for (int i = 0; i < 16; i++)
            s += abs((int)a[j*as+i] - (int)b[j*bs+i]);
    return s;
}

__global__ void downsample4(const uint8_t *__restrict__ src, int sw, int sh,
                            uint8_t *__restrict__ dst, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int sx = x * 4, sy = y * 4;
    int sum = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int px = min(sx + i, sw - 1);
            int py = min(sy + j, sh - 1);
            sum += src[py * sw + px];
        }
    dst[y * dw + x] = (uint8_t)((sum + 8) >> 4);
}

__device__ __forceinline__ void insert_topk(Cand *best, int x, int y, uint32_t sad) {
    if (sad >= best[K-1].sad) return;
    int i = K - 1;
    while (i > 0 && sad < best[i-1].sad) {
        best[i] = best[i-1];
        i--;
    }
    best[i].x = (int16_t)x;
    best[i].y = (int16_t)y;
    best[i].sad = sad;
}

__global__ void coarse_search(const uint8_t *__restrict__ cur, int cw, int ch,
                              const uint8_t *__restrict__ ref, int rw, int rh,
                              int mbw, int mbh, int range, Cand *out) {
    int mb = blockIdx.x * blockDim.x + threadIdx.x;
    if (mb >= mbw * mbh) return;
    int mx = mb % mbw, my = mb / mbw;
    int bx = mx * 4, by = my * 4;
    Cand best[K];
    #pragma unroll
    for (int i = 0; i < K; i++) { best[i].x = 0; best[i].y = 0; best[i].sad = 0xFFFFFFFFu; }
    int r = max(1, range / 4);
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = max(0, min(bx + dx, cw - 4));
            int y = max(0, min(by + dy, ch - 4));
            uint32_t sad = sad4x4(cur + by * cw + bx, cw, ref + y * rw + x, rw);
            // MV relative to the MB origin at quarter resolution -> full resolution
            insert_topk(best, dx * 4, dy * 4, sad);
        }
    }
    for (int i = 0; i < K; i++) out[mb * K + i] = best[i];
}

__global__ void refine_search(const uint8_t *__restrict__ cur, int cw, int ch,
                              const uint8_t *__restrict__ ref, int rw, int rh,
                              int mbw, int mbh, int radius, const Cand *coarse, Cand *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mbw * mbh * K;
    if (idx >= total) return;
    int mb = idx / K;
    int mx = mb % mbw, my = mb / mbw;
    int bx = mx * 16, by = my * 16;
    int cx = bx + coarse[idx].x, cy = by + coarse[idx].y;
    Cand best; best.x = 0; best.y = 0; best.sad = 0xFFFFFFFFu;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int x = max(0, min(cx + dx, cw - 16));
            int y = max(0, min(cy + dy, ch - 16));
            uint32_t sad = sad16_scalar(cur + by * cw + bx, cw, ref + y * rw + x, rw);
            if (sad < best.sad) { best.sad = sad; best.x = (int16_t)(x - bx); best.y = (int16_t)(y - by); }
        }
    }
    out[idx] = best;
}

// Sort K candidates per MB and keep the best four (host side).
static int cmp_cand(const void *a, const void *b) {
    uint32_t sa = ((const Cand *)a)->sad, sb = ((const Cand *)b)->sad;
    return sa < sb ? -1 : sa > sb;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s width height frames.yuv out.bin [range] [radius]\n", argv[0]);
        return 1;
    }
    int w = atoi(argv[1]), h = atoi(argv[2]);
    int range = argc > 5 ? atoi(argv[5]) : 16;
    int radius = argc > 6 ? atoi(argv[6]) : 3;
    int mbw = (w + 15) / 16, mbh = (h + 15) / 16;
    size_t fsz = (size_t)w * h;
    FILE *f = fopen(argv[3], "rb");
    if (!f) { perror(argv[3]); return 1; }
    uint8_t *hcur = (uint8_t *)malloc(fsz), *href = (uint8_t *)malloc(fsz);
    if (fread(hcur, 1, fsz, f) != fsz || fread(href, 1, fsz, f) != fsz) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    int reps = 20;
    uint8_t *dcur, *dref, *dc_low, *dr_low;
    Cand *dcoarse, *dout;
    Cand *hout;
    ck(cudaMalloc(&dcur, fsz), "cur"); ck(cudaMalloc(&dref, fsz), "ref");
    ck(cudaMalloc(&dc_low, (w/4+1)*(h/4+1)), "clow"); ck(cudaMalloc(&dr_low, (w/4+1)*(h/4+1)), "rlow");
    ck(cudaMalloc(&dcoarse, (size_t)mbw*mbh*K*sizeof(Cand)), "coarse");
    ck(cudaMalloc(&dout, (size_t)mbw*mbh*K*sizeof(Cand)), "out");
    ck(cudaMallocHost(&hout, (size_t)mbw*mbh*K*sizeof(Cand)), "pinned");
    cudaStream_t st; ck(cudaStreamCreate(&st), "stream");
    cudaEvent_t e0, e1; ck(cudaEventCreate(&e0), "e0"); ck(cudaEventCreate(&e1), "e1");

    ck(cudaMemcpyAsync(dcur, hcur, fsz, cudaMemcpyHostToDevice, st), "copy cur");
    ck(cudaMemcpyAsync(dref, href, fsz, cudaMemcpyHostToDevice, st), "copy ref");
    ck(cudaStreamSynchronize(st), "sync copy");

    dim3 blk(16, 16);
    dim3 grd((w/4 + 15)/16, (h/4 + 15)/16);
    ck(cudaEventRecord(e0, st), "rec0");
    for (int r = 0; r < reps; r++) {
        downsample4<<<grd, blk, 0, st>>>(dcur, w, h, dc_low, w/4, h/4);
        downsample4<<<grd, blk, 0, st>>>(dref, w, h, dr_low, w/4, h/4);
        int threads = 256;
        coarse_search<<<(mbw*mbh + threads - 1)/threads, threads, 0, st>>>(dc_low, w/4, h/4, dr_low, w/4, h/4, mbw, mbh, range, dcoarse);
        int total = mbw*mbh*K;
        refine_search<<<(total + threads - 1)/threads, threads, 0, st>>>(dcur, w, h, dref, w, h, mbw, mbh, radius, dcoarse, dout);
        ck(cudaMemcpyAsync(hout, dout, (size_t)total*sizeof(Cand), cudaMemcpyDeviceToHost, st), "copy out");
    }
    ck(cudaEventRecord(e1, st), "rec1");
    ck(cudaStreamSynchronize(st), "sync");
    float ms = 0; ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    printf("%dx%d mb=%dx%d range=%d radius=%d reps=%d: %.3f ms/frame total, %.3f ms/search\n",
           w, h, mbw, mbh, range, radius, reps, ms/reps, ms/reps);

    // Write candidates of the final frame for inspection.
    FILE *o = fopen(argv[4], "wb");
    if (o) {
        fwrite(hout, sizeof(Cand), (size_t)mbw*mbh*K, o);
        fclose(o);
    }

    // CPU sort of each MB's K candidates and a quality summary: best SAD over K.
    uint64_t sum = 0;
    for (int mb = 0; mb < mbw*mbh; mb++) {
        qsort(hout + mb*K, K, sizeof(Cand), cmp_cand);
        sum += hout[mb*K].sad;
    }
    printf("mean best SAD: %.2f\n", (double)sum / (mbw*mbh));
    ck(cudaFreeHost(hout), "free pinned");
    ck(cudaFree(dcur), "f1"); ck(cudaFree(dref), "f2"); ck(cudaFree(dc_low), "f3"); ck(cudaFree(dr_low), "f4");
    ck(cudaFree(dcoarse), "f5"); ck(cudaFree(dout), "f6");
    free(hcur); free(href);
    return 0;
}
