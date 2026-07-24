#include "bf_cuda_bev.h"
#include "bf_cuda_ops.h"

#include <cuda_runtime.h>
#ifdef BF_CUDA_VENDOR
#include <cudnn.h>
#endif

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct gpu_conv {
    float *weight, *bias;
    size_t weight_bytes, bias_bytes, workspace_bytes;
    int ci, co, ih, iw, oh, ow, kernel, stride, padding, transpose, relu;
#ifdef BF_CUDA_VENDOR
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    cudnnConvolutionFwdAlgo_t forward_algo;
    cudnnConvolutionBwdDataAlgo_t backward_algo;
#endif
};

struct bf_cuda_bev_stage {
#ifdef BF_CUDA_VENDOR
    cudnnHandle_t handle;
#endif
    gpu_conv layers[18];
    size_t layer_count, height, width, resident_bytes, workspace_bytes;
    float *scratch_a, *scratch_b;
    void *workspace;
    void *stream;
};

static int fail(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list args; va_start(args, format);
        std::vsnprintf(error, cap, format, args); va_end(args);
    }
    return 0;
}

static int cuda_ok(cudaError_t status, char *error, size_t cap, const char *where) {
    return status == cudaSuccess ? 1 : fail(error, cap, "%s: %s", where,
                                             cudaGetErrorString(status));
}

#ifdef BF_CUDA_VENDOR
static int cudnn_ok(cudnnStatus_t status, char *error, size_t cap, const char *where) {
    return status == CUDNN_STATUS_SUCCESS ? 1 : fail(error, cap, "%s: %s", where,
                                                      cudnnGetErrorString(status));
}
#endif

static const float *tensor_f32(const bf_model *model, const char *name,
                               uint32_t rank, const uint32_t *dims,
                               char *error, size_t cap) {
    const bf_tensor *tensor = bf_model_find(model, name);
    if (!tensor || tensor->dtype != BF_DTYPE_F32 || tensor->rank != rank) {
        fail(error, cap, "%s contract mismatch", name); return nullptr;
    }
    for (uint32_t i = 0; i < rank; ++i)
        if (tensor->dims[i] != dims[i]) {
            fail(error, cap, "%s shape mismatch", name); return nullptr;
        }
    return (const float *)tensor->data;
}

static int upload(float **device, const float *host, size_t count,
                  size_t *resident, char *error, size_t cap) {
    size_t bytes = count * sizeof(float);
    if (!cuda_ok(cudaMalloc((void **)device, bytes), error, cap, "cudaMalloc weight") ||
        !cuda_ok(cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice),
                 error, cap, "upload weight")) return 0;
    *resident += bytes;
    return 1;
}

static int bind_folded(bf_cuda_bev_stage *stage, gpu_conv *layer,
    const bf_model *model, const char *weight_name, const char *bias_name,
    const char *bn_prefix, int ci, int co, int kernel,
    int ih, int iw, int stride, int padding, int transpose, int relu,
    char *error, size_t cap) {
    uint32_t wdims[4] = {(uint32_t)(transpose ? ci : co),
                         (uint32_t)(transpose ? co : ci),
                         (uint32_t)kernel, (uint32_t)kernel};
    const float *source_weight = tensor_f32(model, weight_name, 4, wdims, error, cap);
    if (!source_weight) return 0;
    size_t weight_count = (size_t)ci * co * kernel * kernel;
    float *weight = (float *)std::malloc(weight_count * sizeof(float));
    float *bias = (float *)std::malloc((size_t)co * sizeof(float));
    if (!weight || !bias) { std::free(weight); std::free(bias); return fail(error, cap, "fold buffer allocation failed"); }
    std::memcpy(weight, source_weight, weight_count * sizeof(float));
    if (bn_prefix) {
        const char *suffix[4] = {"weight", "bias", "running_mean", "running_var"};
        const float *bn[4] = {nullptr, nullptr, nullptr, nullptr};
        uint32_t dims[1] = {(uint32_t)co};
        char name[160];
        for (int i = 0; i < 4; ++i) {
            std::snprintf(name, sizeof(name), "%s.%s", bn_prefix, suffix[i]);
            bn[i] = tensor_f32(model, name, 1, dims, error, cap);
            if (!bn[i]) { std::free(weight); std::free(bias); return 0; }
        }
        for (int o = 0; o < co; ++o) {
            float factor = bn[0][o] / std::sqrt(bn[3][o] + 1e-3f);
            bias[o] = bn[1][o] - bn[2][o] * factor;
            if (!transpose) {
                size_t begin = (size_t)o * ci * kernel * kernel;
                for (size_t j = 0; j < (size_t)ci * kernel * kernel; ++j)
                    weight[begin + j] *= factor;
            } else {
                for (int in = 0; in < ci; ++in)
                    for (int ky = 0; ky < kernel; ++ky)
                        for (int kx = 0; kx < kernel; ++kx)
                            weight[(((size_t)in * co + o) * kernel + ky) * kernel + kx] *= factor;
            }
        }
    } else {
        uint32_t dims[1] = {(uint32_t)co};
        const float *source_bias = tensor_f32(model, bias_name, 1, dims, error, cap);
        if (!source_bias) { std::free(weight); std::free(bias); return 0; }
        std::memcpy(bias, source_bias, (size_t)co * sizeof(float));
    }
    layer->ci = ci; layer->co = co; layer->ih = ih; layer->iw = iw;
    layer->kernel = kernel; layer->stride = stride; layer->padding = padding;
    layer->transpose = transpose; layer->relu = relu;
    layer->oh = transpose ? (ih - 1) * stride - 2 * padding + kernel
                          : (ih + 2 * padding - kernel) / stride + 1;
    layer->ow = transpose ? (iw - 1) * stride - 2 * padding + kernel
                          : (iw + 2 * padding - kernel) / stride + 1;
    if (!upload(&layer->weight, weight, weight_count, &stage->resident_bytes, error, cap) ||
        !upload(&layer->bias, bias, (size_t)co, &stage->resident_bytes, error, cap)) {
        std::free(weight); std::free(bias); return 0;
    }
    layer->weight_bytes = weight_count * sizeof(float);
    layer->bias_bytes = (size_t)co * sizeof(float);
    std::free(weight); std::free(bias);
#ifdef BF_CUDA_VENDOR
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&layer->input_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateTensorDescriptor(&layer->output_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateTensorDescriptor(&layer->bias_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateFilterDescriptor(&layer->filter_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateConvolutionDescriptor(&layer->conv_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateActivationDescriptor(&layer->activation_desc);
    if (!cudnn_ok(status, error, cap, "create cuDNN descriptor")) return 0;
    if (!cudnn_ok(cudnnSetTensor4dDescriptor(layer->input_desc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, ci, ih, iw), error, cap, "set input descriptor") ||
        !cudnn_ok(cudnnSetTensor4dDescriptor(layer->output_desc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, co, layer->oh, layer->ow), error, cap, "set output descriptor") ||
        !cudnn_ok(cudnnSetTensor4dDescriptor(layer->bias_desc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, co, 1, 1), error, cap, "set bias descriptor") ||
        !cudnn_ok(cudnnSetFilter4dDescriptor(layer->filter_desc, CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW, transpose ? ci : co, transpose ? co : ci,
            kernel, kernel), error, cap, "set filter descriptor") ||
        !cudnn_ok(cudnnSetConvolution2dDescriptor(layer->conv_desc, padding, padding,
            stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
            error, cap, "set convolution descriptor") ||
        !cudnn_ok(cudnnSetConvolutionMathType(layer->conv_desc, CUDNN_FMA_MATH),
            error, cap, "set strict convolution math") ||
        !cudnn_ok(cudnnSetActivationDescriptor(layer->activation_desc,
            relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY,
            CUDNN_NOT_PROPAGATE_NAN, 0.0), error, cap, "set activation descriptor")) return 0;
    int returned = 0;
    if (!transpose) {
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        if (!cudnn_ok(cudnnGetConvolutionForwardAlgorithm_v7(stage->handle,
                layer->input_desc, layer->filter_desc, layer->conv_desc,
                layer->output_desc, 8, &returned, perf), error, cap, "select forward algorithm") || !returned)
            return 0;
        int selected = -1;
        for (int i = 0; i < returned; ++i)
            if (perf[i].status == CUDNN_STATUS_SUCCESS &&
                perf[i].determinism == CUDNN_DETERMINISTIC) { selected = i; break; }
        if (selected < 0) return fail(error, cap, "no deterministic cuDNN forward algorithm");
        layer->forward_algo = perf[selected].algo;
        if (!cudnn_ok(cudnnGetConvolutionForwardWorkspaceSize(stage->handle,
                layer->input_desc, layer->filter_desc, layer->conv_desc,
                layer->output_desc, layer->forward_algo, &layer->workspace_bytes),
                error, cap, "query forward workspace")) return 0;
    } else {
        cudnnConvolutionBwdDataAlgoPerf_t perf[8];
        if (!cudnn_ok(cudnnGetConvolutionBackwardDataAlgorithm_v7(stage->handle,
                layer->filter_desc, layer->input_desc, layer->conv_desc,
                layer->output_desc, 8, &returned, perf), error, cap, "select transpose algorithm") || !returned)
            return 0;
        int selected = -1;
        for (int i = 0; i < returned; ++i)
            if (perf[i].status == CUDNN_STATUS_SUCCESS &&
                perf[i].determinism == CUDNN_DETERMINISTIC) { selected = i; break; }
        if (selected < 0) return fail(error, cap, "no deterministic cuDNN transpose algorithm");
        layer->backward_algo = perf[selected].algo;
        if (!cudnn_ok(cudnnGetConvolutionBackwardDataWorkspaceSize(stage->handle,
                layer->filter_desc, layer->input_desc, layer->conv_desc,
                layer->output_desc, layer->backward_algo, &layer->workspace_bytes),
                error, cap, "query transpose workspace")) return 0;
    }
    if (layer->workspace_bytes > stage->workspace_bytes)
        stage->workspace_bytes = layer->workspace_bytes;
#endif
    return 1;
}

static void destroy_layer(gpu_conv *layer) {
    cudaFree(layer->weight); cudaFree(layer->bias);
#ifdef BF_CUDA_VENDOR
    if (layer->input_desc) cudnnDestroyTensorDescriptor(layer->input_desc);
    if (layer->output_desc) cudnnDestroyTensorDescriptor(layer->output_desc);
    if (layer->bias_desc) cudnnDestroyTensorDescriptor(layer->bias_desc);
    if (layer->filter_desc) cudnnDestroyFilterDescriptor(layer->filter_desc);
    if (layer->conv_desc) cudnnDestroyConvolutionDescriptor(layer->conv_desc);
    if (layer->activation_desc) cudnnDestroyActivationDescriptor(layer->activation_desc);
#endif
}

extern "C" int bf_cuda_bev_stage_create(const bf_model *model, size_t height,
    size_t width, bf_cuda_bev_stage **out, char *error, size_t cap) {
    if (out) *out = nullptr;
    if (!model || !out || !height || !width || (height & 1) || (width & 1) ||
        height > INT_MAX || width > INT_MAX)
        return fail(error, cap, "invalid CUDA BEV stage contract");
    bf_cuda_bev_stage *s = (bf_cuda_bev_stage *)std::calloc(1, sizeof(*s));
    if (!s) return fail(error, cap, "CUDA BEV context allocation failed");
    const int conv_index[6] = {1,4,7,10,13,16};
    const int bn_index[6] = {2,5,8,11,14,17};
    char weight[160], bn[160];
    s->height = height; s->width = width;
#ifdef BF_CUDA_VENDOR
    if (!cudnn_ok(cudnnCreate(&s->handle), error, cap, "create cuDNN handle")) goto failure;
#endif
#define ADD(w,b,bn,ci,co,k,ih,iw,st,p,tr,r) do { \
    if (!bind_folded(s, &s->layers[s->layer_count++], model, w, b, bn, ci, co, k, \
                     ih, iw, st, p, tr, r, error, cap)) goto failure; \
} while (0)
    ADD("fuser.conv.0.weight", nullptr, "fuser.conv.1", 336,256,3,height,width,1,1,0,1);
    for (int layer = 0; layer < 6; ++layer) {
        std::snprintf(weight,sizeof(weight),"backbone_2d.blocks.0.%d.weight",conv_index[layer]);
        std::snprintf(bn,sizeof(bn),"backbone_2d.blocks.0.%d",bn_index[layer]);
        ADD(weight,nullptr,bn,layer?128:256,128,3,height,width,1,1,0,1);
    }
    ADD("backbone_2d.deblocks.0.0.weight",nullptr,"backbone_2d.deblocks.0.1",128,256,1,height,width,1,0,0,1);
    for (int layer = 0; layer < 6; ++layer) {
        std::snprintf(weight,sizeof(weight),"backbone_2d.blocks.1.%d.weight",conv_index[layer]);
        std::snprintf(bn,sizeof(bn),"backbone_2d.blocks.1.%d",bn_index[layer]);
        ADD(weight,nullptr,bn,layer?256:128,256,3,layer?height/2:height,
            layer?width/2:width,layer?1:2,1,0,1);
    }
    ADD("backbone_2d.deblocks.1.0.weight",nullptr,"backbone_2d.deblocks.1.1",256,256,2,height/2,width/2,2,0,1,1);
    ADD("dense_head.shared_conv.weight","dense_head.shared_conv.bias",nullptr,512,128,3,height,width,1,1,0,0);
    ADD("dense_head.heatmap_head.0.conv.weight",nullptr,"dense_head.heatmap_head.0.bn",128,128,3,height,width,1,1,0,1);
    ADD("dense_head.heatmap_head.1.weight","dense_head.heatmap_head.1.bias",nullptr,128,10,3,height,width,1,1,0,0);
#undef ADD
    { size_t scratch_bytes = 256 * height * width * sizeof(float);
      if (!cuda_ok(cudaMalloc((void **)&s->scratch_a, scratch_bytes), error, cap, "allocate BEV scratch A") ||
          !cuda_ok(cudaMalloc((void **)&s->scratch_b, scratch_bytes), error, cap, "allocate BEV scratch B")) goto failure;
      s->resident_bytes += 2 * scratch_bytes; }
    if (s->workspace_bytes && !cuda_ok(cudaMalloc(&s->workspace, s->workspace_bytes),
                                        error, cap, "allocate cuDNN workspace")) goto failure;
    s->resident_bytes += s->workspace_bytes;
    *out = s; return 1;
failure:
    bf_cuda_bev_stage_destroy(s); return 0;
}

extern "C" void bf_cuda_bev_stage_destroy(bf_cuda_bev_stage *s) {
    if (!s) return;
    for (size_t i = 0; i < s->layer_count; ++i) destroy_layer(&s->layers[i]);
    cudaFree(s->scratch_a); cudaFree(s->scratch_b); cudaFree(s->workspace);
#ifdef BF_CUDA_VENDOR
    if (s->handle) cudnnDestroy(s->handle);
#endif
    std::free(s);
}

static int execute(bf_cuda_bev_stage *s, gpu_conv *l, const float *input,
                   float *output, char *error, size_t cap) {
#ifdef BF_CUDA_VENDOR
    const float one = 1.0f, zero = 0.0f;
    cudnnStatus_t status;
    if (!l->transpose) {
        status = cudnnConvolutionBiasActivationForward(s->handle, &one,
            l->input_desc, input, l->filter_desc, l->weight, l->conv_desc,
            l->forward_algo, s->workspace, s->workspace_bytes, &zero,
            l->output_desc, output, l->bias_desc, l->bias,
            l->activation_desc, l->output_desc, output);
    } else {
        status = cudnnConvolutionBackwardData(s->handle, &one, l->filter_desc,
            l->weight, l->input_desc, input, l->conv_desc, l->backward_algo,
            s->workspace, s->workspace_bytes, &zero, l->output_desc, output);
        if (status == CUDNN_STATUS_SUCCESS)
            status = cudnnAddTensor(s->handle, &one, l->bias_desc, l->bias,
                                    &one, l->output_desc, output);
        if (status == CUDNN_STATUS_SUCCESS && l->relu)
            status = cudnnActivationForward(s->handle, l->activation_desc, &one,
                l->output_desc, output, &zero, l->output_desc, output);
    }
    return cudnn_ok(status, error, cap, "execute CUDA BEV convolution");
#else
    bf_cuda_conv2d_desc desc = {
        1, l->ci, l->co, l->ih, l->iw, l->kernel, l->kernel,
        l->stride, l->stride, l->padding, l->padding, l->oh, l->ow,
        l->transpose, l->relu
    };
    return bf_cuda_conv2d_f32(&desc, input, l->weight, l->bias, output,
                              s->stream, error, cap);
#endif
}

extern "C" int bf_cuda_bev_stage_forward(bf_cuda_bev_stage *s,
    const float *input, float *spatial, float *shared, float *heatmap,
    void *stream_value, char *error, size_t cap) {
    if (!s || !input || !spatial || !shared || !heatmap)
        return fail(error, cap, "invalid CUDA BEV forward buffers");
#ifdef BF_CUDA_VENDOR
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
    if (!cudnn_ok(cudnnSetStream(s->handle, stream), error, cap, "set cuDNN stream")) return 0;
#else
    s->stream = stream_value;
#endif
    size_t hw = s->height * s->width;
    size_t i = 0;
    if (!execute(s,&s->layers[i++],input,s->scratch_a,error,cap)) return 0;
    const float *current = s->scratch_a; float *next = s->scratch_b;
    for (int layer=0; layer<6; ++layer) {
        if (!execute(s,&s->layers[i++],current,next,error,cap)) return 0;
        current=next; next=next==s->scratch_a?s->scratch_b:s->scratch_a;
    }
    if (!execute(s,&s->layers[i++],current,spatial,error,cap)) return 0;
    next = current==s->scratch_a?s->scratch_b:s->scratch_a;
    for (int layer=0; layer<6; ++layer) {
        if (!execute(s,&s->layers[i++],current,next,error,cap)) return 0;
        current=next; next=next==s->scratch_a?s->scratch_b:s->scratch_a;
    }
    if (!execute(s,&s->layers[i++],current,spatial+256*hw,error,cap) ||
        !execute(s,&s->layers[i++],spatial,shared,error,cap) ||
        !execute(s,&s->layers[i++],shared,s->scratch_a,error,cap) ||
        !execute(s,&s->layers[i++],s->scratch_a,heatmap,error,cap)) return 0;
    return i == s->layer_count;
}

extern "C" size_t bf_cuda_bev_stage_resident_bytes(const bf_cuda_bev_stage *s) {
    return s ? s->resident_bytes : 0;
}
