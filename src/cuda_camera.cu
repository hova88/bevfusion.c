#include "bf_cuda_camera.h"
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
#include <climits>

struct camera_conv {
    float *weight, *bias;
    int n, ci, co, ih, iw, oh, ow, kernel, stride, padding, relu;
    size_t workspace_bytes, weight_bytes, bias_bytes;
#ifdef BF_CUDA_VENDOR
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnActivationDescriptor_t activation_desc;
    cudnnConvolutionFwdAlgo_t algorithm;
#endif
};

struct bf_cuda_camera_neck {
#ifdef BF_CUDA_VENDOR
    cudnnHandle_t handle;
#endif
    camera_conv layers[13];
    size_t layer_count, batches, h, w, h1, w1, h2, w2, bev_h, bev_w;
    float *scratch_a, *scratch_b, *scratch_c;
    size_t scratch_a_bytes, scratch_b_bytes, scratch_c_bytes;
    void *workspace;
    void *stream;
    size_t workspace_bytes, resident_bytes;
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
    if (!tensor || tensor->dtype != BF_DTYPE_F32 || tensor->rank != rank)
        return fail(error, cap, "%s contract mismatch", name), nullptr;
    for (uint32_t i = 0; i < rank; ++i)
        if (tensor->dims[i] != dims[i])
            return fail(error, cap, "%s shape mismatch", name), nullptr;
    return (const float *)tensor->data;
}

static int upload(float **device, const float *host, size_t count,
                  size_t *resident, char *error, size_t cap) {
    size_t bytes = count * sizeof(float);
    if (!cuda_ok(cudaMalloc((void **)device, bytes), error, cap, "allocate camera weight") ||
        !cuda_ok(cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice),
                 error, cap, "upload camera weight")) return 0;
    *resident += bytes;
    return 1;
}

static int bind_folded(bf_cuda_camera_neck *neck, camera_conv *layer,
    const bf_model *model, const char *weight_name, const char *bias_name,
    const char *bn_prefix, int n, int ci, int co, int kernel,
    int ih, int iw, int stride, int padding, int relu, float epsilon,
    char *error, size_t cap) {
    uint32_t wdims[4] = {(uint32_t)co, (uint32_t)ci,
                         (uint32_t)kernel, (uint32_t)kernel};
    const float *source_weight = tensor_f32(model, weight_name, 4, wdims, error, cap);
    if (!source_weight) return 0;
    size_t weight_count = (size_t)co * ci * kernel * kernel;
    float *weight = (float *)std::malloc(weight_count * sizeof(float));
    float *bias = (float *)std::calloc((size_t)co, sizeof(float));
    if (!weight || !bias) {
        std::free(weight); std::free(bias);
        return fail(error, cap, "camera fold allocation failed");
    }
    std::memcpy(weight, source_weight, weight_count * sizeof(float));
    uint32_t bdims[1] = {(uint32_t)co};
    if (bias_name) {
        const float *source_bias = tensor_f32(model, bias_name, 1, bdims, error, cap);
        if (!source_bias) { std::free(weight); std::free(bias); return 0; }
        std::memcpy(bias, source_bias, (size_t)co * sizeof(float));
    }
    if (bn_prefix) {
        const char *suffix[4] = {"weight", "bias", "running_mean", "running_var"};
        const float *bn[4] = {nullptr, nullptr, nullptr, nullptr};
        char name[192];
        for (int i = 0; i < 4; ++i) {
            std::snprintf(name, sizeof(name), "%s.%s", bn_prefix, suffix[i]);
            bn[i] = tensor_f32(model, name, 1, bdims, error, cap);
            if (!bn[i]) { std::free(weight); std::free(bias); return 0; }
        }
        size_t kernel_span = (size_t)ci * kernel * kernel;
        for (int output = 0; output < co; ++output) {
            float factor = bn[0][output] / std::sqrt(bn[3][output] + epsilon);
            bias[output] = (bias[output] - bn[2][output]) * factor + bn[1][output];
            for (size_t i = 0; i < kernel_span; ++i)
                weight[(size_t)output * kernel_span + i] *= factor;
        }
    }
    layer->n=n; layer->ci=ci; layer->co=co; layer->ih=ih; layer->iw=iw;
    layer->kernel=kernel; layer->stride=stride; layer->padding=padding; layer->relu=relu;
    layer->oh=(ih+2*padding-kernel)/stride+1;
    layer->ow=(iw+2*padding-kernel)/stride+1;
    if (!upload(&layer->weight, weight, weight_count, &neck->resident_bytes, error, cap) ||
        !upload(&layer->bias, bias, (size_t)co, &neck->resident_bytes, error, cap)) {
        std::free(weight); std::free(bias); return 0;
    }
    layer->weight_bytes=weight_count*sizeof(float); layer->bias_bytes=(size_t)co*sizeof(float);
    std::free(weight); std::free(bias);
#ifdef BF_CUDA_VENDOR
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&layer->input_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateTensorDescriptor(&layer->output_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateTensorDescriptor(&layer->bias_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateFilterDescriptor(&layer->filter_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateConvolutionDescriptor(&layer->conv_desc);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnCreateActivationDescriptor(&layer->activation_desc);
    if (!cudnn_ok(status,error,cap,"create camera cuDNN descriptor") ||
        !cudnn_ok(cudnnSetTensor4dDescriptor(layer->input_desc,CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,n,ci,ih,iw),error,cap,"set camera input") ||
        !cudnn_ok(cudnnSetTensor4dDescriptor(layer->output_desc,CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,n,co,layer->oh,layer->ow),error,cap,"set camera output") ||
        !cudnn_ok(cudnnSetTensor4dDescriptor(layer->bias_desc,CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,1,co,1,1),error,cap,"set camera bias") ||
        !cudnn_ok(cudnnSetFilter4dDescriptor(layer->filter_desc,CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,co,ci,kernel,kernel),error,cap,"set camera filter") ||
        !cudnn_ok(cudnnSetConvolution2dDescriptor(layer->conv_desc,padding,padding,
            stride,stride,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT),
            error,cap,"set camera convolution") ||
        !cudnn_ok(cudnnSetConvolutionMathType(layer->conv_desc,CUDNN_FMA_MATH),
            error,cap,"set strict camera math") ||
        !cudnn_ok(cudnnSetActivationDescriptor(layer->activation_desc,
            relu?CUDNN_ACTIVATION_RELU:CUDNN_ACTIVATION_IDENTITY,
            CUDNN_NOT_PROPAGATE_NAN,0.0),error,cap,"set camera activation")) return 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8]; int returned=0, selected=-1;
    if (!cudnn_ok(cudnnGetConvolutionForwardAlgorithm_v7(neck->handle,
        layer->input_desc,layer->filter_desc,layer->conv_desc,layer->output_desc,
        8,&returned,perf),error,cap,"select camera algorithm")) return 0;
    /* The unconstrained first deterministic cuDNN choice is both slower and
     * over 230 MiB larger on the target Ada GPU. */
    size_t workspace_limit=16ull*1024*1024;
    const char *limit_text=std::getenv("BF_CUDA_CAMERA_WORKSPACE_MIB");
    if(limit_text) {
        char *end=nullptr; unsigned long long mib=std::strtoull(limit_text,&end,10);
        if(end&&!*end&&mib<=SIZE_MAX/(1024ull*1024ull))
            workspace_limit=(size_t)mib*1024*1024;
    }
    for (int i=0;i<returned;++i)
        if (perf[i].status==CUDNN_STATUS_SUCCESS &&
            perf[i].determinism==CUDNN_DETERMINISTIC &&
            perf[i].memory<=workspace_limit) { selected=i; break; }
    if (selected<0) return fail(error,cap,"no deterministic camera convolution");
    layer->algorithm=perf[selected].algo;
    if (!cudnn_ok(cudnnGetConvolutionForwardWorkspaceSize(neck->handle,
        layer->input_desc,layer->filter_desc,layer->conv_desc,layer->output_desc,
        layer->algorithm,&layer->workspace_bytes),error,cap,"query camera workspace")) return 0;
    if (layer->workspace_bytes>neck->workspace_bytes) neck->workspace_bytes=layer->workspace_bytes;
#endif
    return 1;
}

static void destroy_layer(camera_conv *layer) {
    cudaFree(layer->weight); cudaFree(layer->bias);
#ifdef BF_CUDA_VENDOR
    if(layer->input_desc)cudnnDestroyTensorDescriptor(layer->input_desc);
    if(layer->output_desc)cudnnDestroyTensorDescriptor(layer->output_desc);
    if(layer->bias_desc)cudnnDestroyTensorDescriptor(layer->bias_desc);
    if(layer->filter_desc)cudnnDestroyFilterDescriptor(layer->filter_desc);
    if(layer->conv_desc)cudnnDestroyConvolutionDescriptor(layer->conv_desc);
    if(layer->activation_desc)cudnnDestroyActivationDescriptor(layer->activation_desc);
#endif
}

static int execute(bf_cuda_camera_neck *neck, camera_conv *layer,
                   const float *input, float *output,
                   char *error, size_t cap) {
#ifdef BF_CUDA_VENDOR
    const float one=1.0f, zero=0.0f;
    cudnnStatus_t status=cudnnConvolutionBiasActivationForward(neck->handle,&one,
        layer->input_desc,input,layer->filter_desc,layer->weight,layer->conv_desc,
        layer->algorithm,neck->workspace,neck->workspace_bytes,&zero,
        layer->output_desc,output,layer->bias_desc,layer->bias,
        layer->activation_desc,layer->output_desc,output);
    return cudnn_ok(status,error,cap,"execute camera convolution");
#else
    bf_cuda_conv2d_desc desc = {
        layer->n, layer->ci, layer->co, layer->ih, layer->iw,
        layer->kernel, layer->kernel, layer->stride, layer->stride,
        layer->padding, layer->padding, layer->oh, layer->ow, 0, layer->relu
    };
    return bf_cuda_conv2d_f32(&desc, input, layer->weight, layer->bias,
                              output, neck->stream, error, cap);
#endif
}

__global__ static void fpn_concat_kernel(
    const float *__restrict__ lower, const float *__restrict__ upper,
    float *__restrict__ output, int batches, int lower_channels,
    int upper_channels, int oh, int ow, int ih, int iw) {
    size_t count=(size_t)batches*(lower_channels+upper_channels)*oh*ow;
    size_t index=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count)return;
    int x=index%ow; size_t q=index/ow;
    int y=q%oh; q/=oh; int channel=q%(lower_channels+upper_channels);
    int batch=q/(lower_channels+upper_channels);
    if(channel<lower_channels) {
        output[index]=lower[((size_t)batch*lower_channels+channel)*oh*ow+y*ow+x];
        return;
    }
    int c=channel-lower_channels;
    float sy=((float)y+0.5f)*(float)ih/(float)oh-0.5f;
    float sx=((float)x+0.5f)*(float)iw/(float)ow-0.5f;
    sy=fmaxf(sy,0.0f); sx=fmaxf(sx,0.0f);
    int y0=(int)floorf(sy),x0=(int)floorf(sx);
    int y1=min(y0+1,ih-1),x1=min(x0+1,iw-1);
    float fy=sy-y0,fx=sx-x0;
    const float *plane=upper+((size_t)batch*upper_channels+c)*ih*iw;
    float top=plane[y0*iw+x0]*(1.0f-fx)+plane[y0*iw+x1]*fx;
    float bottom=plane[y1*iw+x0]*(1.0f-fx)+plane[y1*iw+x1]*fx;
    output[index]=top*(1.0f-fy)+bottom*fy;
}

__global__ static void depth_concat_kernel(const float *encoded,
    const float *features,float *output,size_t count,int hw) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    size_t q=i/hw; int position=i%hw; int channel=q%320; int batch=q/320;
    output[i]=channel<64?encoded[((size_t)batch*64+channel)*hw+position]
        :features[((size_t)batch*256+channel-64)*hw+position];
}

__global__ static void split_depth_kernel(const float *input,float *logits,
    float *context,size_t count,int hw) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    size_t q=i/hw; int position=i%hw; int channel=q%198; int batch=q/198;
    float value=input[i];
    if(channel<118) logits[((size_t)batch*118+channel)*hw+position]=value;
    else context[((size_t)batch*80+channel-118)*hw+position]=value;
}

__global__ static void transpose_xy_kernel(const float *input,float *output,
    size_t count,int height,int width) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    int y=i%height; size_t q=i/height; int x=q%width; q/=width;
    output[i]=input[(q*height+y)*width+x];
}

extern "C" void bf_cuda_camera_neck_destroy(bf_cuda_camera_neck *neck) {
    if(!neck)return;
    for(size_t i=0;i<neck->layer_count;++i)destroy_layer(&neck->layers[i]);
    cudaFree(neck->scratch_a);cudaFree(neck->scratch_b);cudaFree(neck->scratch_c);
    cudaFree(neck->workspace);
#ifdef BF_CUDA_VENDOR
    if(neck->handle)cudnnDestroy(neck->handle);
#endif
    std::free(neck);
}

extern "C" int bf_cuda_camera_neck_create(const bf_model *model,size_t batches,
    size_t h,size_t w,size_t bev_h,size_t bev_w,bf_cuda_camera_neck **out,
    char *error,size_t cap) {
    if(out)*out=nullptr;
    if(!model||!out||!batches||!h||!w||!bev_h||!bev_w||(bev_h&1)||(bev_w&1)||
       batches>INT_MAX||h>INT_MAX||w>INT_MAX||bev_h>INT_MAX||bev_w>INT_MAX)
        return fail(error,cap,"invalid CUDA camera-neck contract");
    bf_cuda_camera_neck *n=(bf_cuda_camera_neck*)std::calloc(1,sizeof(*n));
    if(!n)return fail(error,cap,"camera-neck host allocation failed");
    n->batches=batches;n->h=h;n->w=w;n->h1=(h+1)/2;n->w1=(w+1)/2;
    n->h2=(n->h1+1)/2;n->w2=(n->w1+1)/2;n->bev_h=bev_h;n->bev_w=bev_w;
#ifdef BF_CUDA_VENDOR
    if(!cudnn_ok(cudnnCreate(&n->handle),error,cap,"create camera cuDNN handle"))goto failure;
#endif
#define ADD(W,B,BN,CI,CO,K,IH,IW,S,P,R) do { \
 if(!bind_folded(n,&n->layers[n->layer_count++],model,W,B,BN,(int)batches,CI,CO,K, \
 (int)(IH),(int)(IW),S,P,R,1e-5f,error,cap))goto failure; } while(0)
    ADD("neck.lateral_convs.1.conv.weight",nullptr,"neck.lateral_convs.1.bn",1152,256,1,n->h1,n->w1,1,0,1);
    ADD("neck.fpn_convs.1.conv.weight",nullptr,"neck.fpn_convs.1.bn",256,256,3,n->h1,n->w1,1,1,1);
    ADD("neck.lateral_convs.0.conv.weight",nullptr,"neck.lateral_convs.0.bn",448,256,1,h,w,1,0,1);
    ADD("neck.fpn_convs.0.conv.weight",nullptr,"neck.fpn_convs.0.bn",256,256,3,h,w,1,1,1);
    ADD("vtransform.dtransform.0.weight","vtransform.dtransform.0.bias","vtransform.dtransform.1",1,8,1,8*h,8*w,1,0,1);
    ADD("vtransform.dtransform.3.weight","vtransform.dtransform.3.bias","vtransform.dtransform.4",8,32,5,8*h,8*w,4,2,1);
    ADD("vtransform.dtransform.6.weight","vtransform.dtransform.6.bias","vtransform.dtransform.7",32,64,5,2*h,2*w,2,2,1);
    ADD("vtransform.depthnet.0.weight","vtransform.depthnet.0.bias","vtransform.depthnet.1",320,256,3,h,w,1,1,1);
    ADD("vtransform.depthnet.3.weight","vtransform.depthnet.3.bias","vtransform.depthnet.4",256,256,3,h,w,1,1,1);
    ADD("vtransform.depthnet.6.weight","vtransform.depthnet.6.bias",nullptr,256,198,1,h,w,1,0,0);
#undef ADD
#define ADD_DOWN(I,J,IH,IW,S) do { char weight[128],bn[128]; \
 std::snprintf(weight,sizeof(weight),"vtransform.downsample.%d.weight",I); \
 std::snprintf(bn,sizeof(bn),"vtransform.downsample.%d",J); \
 if(!bind_folded(n,&n->layers[n->layer_count++],model,weight,nullptr,bn,1,80,80,3, \
 (int)(IH),(int)(IW),S,1,1,1e-5f,error,cap))goto failure; } while(0)
    ADD_DOWN(0,1,bev_h,bev_w,1);ADD_DOWN(3,4,bev_h,bev_w,2);
    ADD_DOWN(6,7,bev_h/2,bev_w/2,1);
#undef ADD_DOWN
    { size_t hw=h*w;
      size_t concat0=batches*448*hw,concat1=batches*1152*n->h1*n->w1;
      size_t depth0=batches*512*hw,bev_full=80*bev_h*bev_w;
      size_t a=concat0>concat1?concat0:concat1;if(depth0>a)a=depth0;if(bev_full>a)a=bev_full;
      size_t b=batches*320*hw,fpn=batches*256*hw;if(fpn>b)b=fpn;
      size_t bev_half=80*(bev_h/2)*(bev_w/2);if(bev_half>b)b=bev_half;
      size_t c=fpn;
      n->scratch_a_bytes=a*sizeof(float);n->scratch_b_bytes=b*sizeof(float);n->scratch_c_bytes=c*sizeof(float);
      if(!cuda_ok(cudaMalloc(&n->scratch_a,n->scratch_a_bytes),error,cap,"allocate camera scratch A")||
         !cuda_ok(cudaMalloc(&n->scratch_b,n->scratch_b_bytes),error,cap,"allocate camera scratch B")||
         !cuda_ok(cudaMalloc(&n->scratch_c,n->scratch_c_bytes),error,cap,"allocate camera scratch C"))goto failure;
      n->resident_bytes+=n->scratch_a_bytes+n->scratch_b_bytes+n->scratch_c_bytes; }
    if(n->workspace_bytes&&!cuda_ok(cudaMalloc(&n->workspace,n->workspace_bytes),error,cap,
                                    "allocate camera cuDNN workspace"))goto failure;
    n->resident_bytes+=n->workspace_bytes;*out=n;return 1;
failure: bf_cuda_camera_neck_destroy(n);return 0;
}

static int set_stream(bf_cuda_camera_neck *n,void *value,char *error,size_t cap) {
#ifdef BF_CUDA_VENDOR
    return cudnn_ok(cudnnSetStream(n->handle,reinterpret_cast<cudaStream_t>(value)),
                    error,cap,"set camera stream");
#else
    n->stream=value; (void)error; (void)cap;
    return 1;
#endif
}

static int fpn_impl(bf_cuda_camera_neck *n,const float *s0,const float *s1,
    const float *s2,float *out0,float *out1,cudaStream_t stream,char *error,size_t cap) {
    size_t count=n->batches*1152*n->h1*n->w1;
    fpn_concat_kernel<<<(unsigned)((count+255)/256),256,0,stream>>>(s1,s2,n->scratch_a,
        (int)n->batches,384,768,(int)n->h1,(int)n->w1,(int)n->h2,(int)n->w2);
    if(!execute(n,&n->layers[0],n->scratch_a,n->scratch_b,error,cap)||
       !execute(n,&n->layers[1],n->scratch_b,out1,error,cap))return 0;
    count=n->batches*448*n->h*n->w;
    fpn_concat_kernel<<<(unsigned)((count+255)/256),256,0,stream>>>(s0,out1,n->scratch_a,
        (int)n->batches,192,256,(int)n->h,(int)n->w,(int)n->h1,(int)n->w1);
    return execute(n,&n->layers[2],n->scratch_a,n->scratch_b,error,cap)&&
           execute(n,&n->layers[3],n->scratch_b,out0,error,cap);
}

extern "C" int bf_cuda_camera_fpn_forward(bf_cuda_camera_neck *n,
    const float *s0,const float *s1,const float *s2,float *out0,float *out1,
    void *stream,char *error,size_t cap) {
    if(!n||!s0||!s1||!s2||!out0||!out1)return fail(error,cap,"invalid CUDA FPN buffers");
    if(!set_stream(n,stream,error,cap))return 0;
    return fpn_impl(n,s0,s1,s2,out0,out1,reinterpret_cast<cudaStream_t>(stream),error,cap);
}

static int depth_impl(bf_cuda_camera_neck *n,const float *features,const float *depth,
    float *logits,float *context,cudaStream_t stream,char *error,size_t cap) {
    size_t hw=n->h*n->w;
    if(!execute(n,&n->layers[4],depth,n->scratch_a,error,cap)||
       !execute(n,&n->layers[5],n->scratch_a,n->scratch_b,error,cap)||
       !execute(n,&n->layers[6],n->scratch_b,n->scratch_a,error,cap))return 0;
    size_t count=n->batches*320*hw;
    depth_concat_kernel<<<(unsigned)((count+255)/256),256,0,stream>>>(n->scratch_a,features,
        n->scratch_b,count,(int)hw);
    if(!execute(n,&n->layers[7],n->scratch_b,n->scratch_a,error,cap)||
       !execute(n,&n->layers[8],n->scratch_a,n->scratch_b,error,cap)||
       !execute(n,&n->layers[9],n->scratch_b,n->scratch_a,error,cap))return 0;
    count=n->batches*198*hw;
    split_depth_kernel<<<(unsigned)((count+255)/256),256,0,stream>>>(n->scratch_a,logits,context,count,(int)hw);
    cudaError_t status=cudaGetLastError();
    return status==cudaSuccess?1:fail(error,cap,"CUDA depth split: %s",cudaGetErrorString(status));
}

extern "C" int bf_cuda_camera_depth_forward(bf_cuda_camera_neck *n,
    const float *features,const float *depth,float *logits,float *context,
    void *stream,char *error,size_t cap) {
    if(!n||!features||!depth||!logits||!context)return fail(error,cap,"invalid CUDA depth buffers");
    if(!set_stream(n,stream,error,cap))return 0;
    return depth_impl(n,features,depth,logits,context,
                      reinterpret_cast<cudaStream_t>(stream),error,cap);
}

extern "C" int bf_cuda_camera_neck_forward(bf_cuda_camera_neck *n,
    const float *s0,const float *s1,const float *s2,const float *depth,
    float *logits,float *context,void *stream,char *error,size_t cap) {
    if(!n||!s0||!s1||!s2||!depth||!logits||!context)
        return fail(error,cap,"invalid CUDA camera-neck buffers");
    if(!set_stream(n,stream,error,cap)||
       !fpn_impl(n,s0,s1,s2,n->scratch_c,n->scratch_c,
                 reinterpret_cast<cudaStream_t>(stream),error,cap))return 0;
    return depth_impl(n,n->scratch_c,depth,logits,context,
                      reinterpret_cast<cudaStream_t>(stream),error,cap);
}

extern "C" int bf_cuda_camera_downsample_forward(bf_cuda_camera_neck *n,
    const float *input,float *output,void *stream,char *error,size_t cap) {
    if(!n||!input||!output)return fail(error,cap,"invalid CUDA downsample buffers");
    if(!set_stream(n,stream,error,cap)||
       !execute(n,&n->layers[10],input,n->scratch_a,error,cap)||
       !execute(n,&n->layers[11],n->scratch_a,n->scratch_b,error,cap)||
       !execute(n,&n->layers[12],n->scratch_b,n->scratch_a,error,cap))return 0;
    size_t count=80*(n->bev_h/2)*(n->bev_w/2);
    transpose_xy_kernel<<<(unsigned)((count+255)/256),256,0,
        reinterpret_cast<cudaStream_t>(stream)>>>(n->scratch_a,output,count,
        (int)(n->bev_h/2),(int)(n->bev_w/2));
    cudaError_t status=cudaGetLastError();
    return status==cudaSuccess?1:fail(error,cap,"CUDA downsample transpose: %s",cudaGetErrorString(status));
}

extern "C" size_t bf_cuda_camera_neck_resident_bytes(const bf_cuda_camera_neck *n) {
    return n?n->resident_bytes:0;
}
