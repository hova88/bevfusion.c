CC ?= cc
NVCC ?= /usr/local/cuda-12.4/bin/nvcc
PYTHON ?= python3
CFLAGS ?= -O3 -march=native -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS += -Iinclude
MODEL ?= bevfusion.bfw
CHECKPOINT ?= ckpts/cbgs_bevfusion.pth
RUNTIME_SOURCES = src/runtime.c src/model.c src/kernels_ref.c src/voxel.c \
	src/swin.c src/swin_backbone.c src/image_fpn.c src/depth_raster.c \
	src/depth_head.c src/lss.c src/lss_downsample.c src/lidar_backbone.c \
	src/bev_stage.c src/transfusion.c src/transfusion_decoder.c
CLI_SOURCES = src/frame.c src/tui.c

.PHONY: all model test portable-test cuda-test clean

all: build/bevfusion

model: $(MODEL)

$(MODEL): tools/export_checkpoint.py $(CHECKPOINT)
	$(PYTHON) tools/export_checkpoint.py $(CHECKPOINT) $@

build/bevfusion: src/main.c $(RUNTIME_SOURCES) $(CLI_SOURCES) include/bevfusion.h include/bf_model.h include/bf_runtime.h include/bf_frame.h include/bf_tui.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) src/main.c $(RUNTIME_SOURCES) $(CLI_SOURCES) -lm -o $@

build/test_model: tests/test_model.c src/model.c include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_model.c src/model.c -o $@

build/kernel_oracle.bfw: tools/oracle_kernels.py tools/export_checkpoint.py
	mkdir -p build
	$(PYTHON) tools/oracle_kernels.py $@

build/depth_raster_oracle.bfw: tools/oracle_depth_raster.py tools/export_checkpoint.py
	mkdir -p build
	$(PYTHON) tools/oracle_depth_raster.py $@

build/bev_stage_oracle.bfw: tools/oracle_bev_stage.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_bev_stage.py $(CHECKPOINT) $@

build/swin_backbone_oracle.bfw: tools/oracle_swin_backbone.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_swin_backbone.py $(CHECKPOINT) $@

build/image_fpn_oracle.bfw: tools/oracle_image_fpn.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_image_fpn.py $(CHECKPOINT) $@

build/depth_head_oracle.bfw: tools/oracle_depth_head.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_depth_head.py $(CHECKPOINT) $@

build/lss_downsample_oracle.bfw: tools/oracle_lss_downsample.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_lss_downsample.py $(CHECKPOINT) $@

build/lidar_backbone_oracle.bfw: tools/oracle_lidar_backbone.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_lidar_backbone.py $(CHECKPOINT) $@

build/transfusion_decoder_oracle.bfw: tools/oracle_transfusion_decoder.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_transfusion_decoder.py $(CHECKPOINT) $@

build/cuda_tail_oracle.bfw: tools/oracle_cuda_tail.py tools/oracle_bev_stage.py tools/oracle_transfusion_decoder.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_cuda_tail.py $(CHECKPOINT) $@

build/cuda_camera_full_oracle.bfw: tools/oracle_cuda_camera_full.py tools/oracle_swin_backbone.py tools/oracle_image_fpn.py tools/oracle_depth_head.py tools/export_checkpoint.py $(CHECKPOINT)
	mkdir -p build
	$(PYTHON) tools/oracle_cuda_camera_full.py $(CHECKPOINT) $@

build/test_kernels: tests/test_kernels.c src/kernels_ref.c src/voxel.c src/model.c include/bf_kernels.h include/bf_voxel.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_kernels.c src/kernels_ref.c src/voxel.c src/model.c -lm -o $@

build/test_depth_raster: tests/test_depth_raster.c src/depth_raster.c src/model.c include/bf_depth_raster.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_depth_raster.c src/depth_raster.c src/model.c -lm -o $@

build/test_lss: tests/test_lss.c src/lss.c src/model.c include/bf_lss.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_lss.c src/lss.c src/model.c -lm -o $@

build/test_swin: tests/test_swin.c src/swin.c src/model.c include/bf_swin.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_swin.c src/swin.c src/model.c -lm -o $@

build/test_transfusion: tests/test_transfusion.c src/transfusion.c src/model.c include/bf_transfusion.h include/bevfusion.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_transfusion.c src/transfusion.c src/model.c -lm -o $@

build/test_bev_stage: tests/test_bev_stage.c src/bev_stage.c src/kernels_ref.c src/model.c include/bf_bev_stage.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_bev_stage.c src/bev_stage.c src/kernels_ref.c src/model.c -lm -o $@

build/test_swin_backbone: tests/test_swin_backbone.c src/swin_backbone.c src/swin.c src/kernels_ref.c src/model.c include/bf_swin_backbone.h include/bf_swin.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_swin_backbone.c src/swin_backbone.c src/swin.c src/kernels_ref.c src/model.c -lm -o $@

build/test_image_fpn: tests/test_image_fpn.c src/image_fpn.c src/kernels_ref.c src/model.c include/bf_image_fpn.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_image_fpn.c src/image_fpn.c src/kernels_ref.c src/model.c -lm -o $@

build/test_depth_head: tests/test_depth_head.c src/depth_head.c src/kernels_ref.c src/model.c include/bf_depth_head.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_depth_head.c src/depth_head.c src/kernels_ref.c src/model.c -lm -o $@

build/test_lss_downsample: tests/test_lss_downsample.c src/lss_downsample.c src/kernels_ref.c src/model.c include/bf_lss_downsample.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_lss_downsample.c src/lss_downsample.c src/kernels_ref.c src/model.c -lm -o $@

build/test_lidar_backbone: tests/test_lidar_backbone.c src/lidar_backbone.c src/kernels_ref.c src/model.c include/bf_lidar_backbone.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_lidar_backbone.c src/lidar_backbone.c src/kernels_ref.c src/model.c -lm -o $@

build/test_transfusion_decoder: tests/test_transfusion_decoder.c src/transfusion_decoder.c src/transfusion.c src/kernels_ref.c src/model.c include/bf_transfusion_decoder.h include/bf_transfusion.h include/bf_kernels.h include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_transfusion_decoder.c src/transfusion_decoder.c src/transfusion.c src/kernels_ref.c src/model.c -lm -o $@

build/test_runtime: tests/test_runtime.c $(RUNTIME_SOURCES) include/bf_runtime.h include/bevfusion.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_runtime.c $(RUNTIME_SOURCES) -lm -o $@

build/test_tui: tests/test_tui.c src/tui.c include/bf_tui.h include/bevfusion.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_tui.c src/tui.c -lm -o $@

build/test_frame: tests/test_frame.c src/frame.c include/bf_frame.h include/bf_runtime.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_frame.c src/frame.c -lm -o $@

build/test_real_voxel: tests/test_real_voxel.c src/frame.c src/voxel.c include/bf_frame.h include/bf_voxel.h
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/test_real_voxel.c src/frame.c src/voxel.c -lm -o $@

build/model_cuda_test.o: src/model.c include/bf_model.h
	mkdir -p build
	$(CC) $(CPPFLAGS) -O2 -std=c11 -c src/model.c -o $@

build/depth_raster_cuda_test.o: src/depth_raster.c include/bf_depth_raster.h
	mkdir -p build
	$(CC) $(CPPFLAGS) -O2 -std=c11 -c src/depth_raster.c -o $@

build/voxel_cuda_test.o: src/voxel.c include/bf_voxel.h
	mkdir -p build
	$(CC) $(CPPFLAGS) -O2 -std=c11 -c src/voxel.c -o $@

build/kernels_cuda_test.o: src/kernels_ref.c include/bf_kernels.h
	mkdir -p build
	$(CC) $(CPPFLAGS) -O2 -std=c11 -c src/kernels_ref.c -o $@

build/frame_cuda_test.o: src/frame.c include/bf_frame.h include/bf_runtime.h
	mkdir -p build
	$(CC) $(CPPFLAGS) -O2 -std=c11 -c src/frame.c -o $@

build/test_cuda_depth_raster: tests/test_cuda_depth_raster.cu src/cuda_depth_raster.cu build/depth_raster_cuda_test.o include/bf_cuda_depth_raster.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_depth_raster.cu src/cuda_depth_raster.cu build/depth_raster_cuda_test.o -o $@

CUDA_RUNTIME_SOURCES = src/cuda_runtime.cu src/cuda_depth_raster.cu src/cuda_voxel.cu src/cuda_lidar.cu src/cuda_swin.cu src/cuda_camera.cu src/cuda_lss.cu src/cuda_bev_stage.cu src/cuda_transfusion.cu
CUDA_C_SOURCES = src/main.c $(RUNTIME_SOURCES) $(CLI_SOURCES)
CUDA_C_OBJECTS = $(patsubst src/%.c,build/cuda_c_%.o,$(CUDA_C_SOURCES))

build/cuda_c_%.o: src/%.c
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) -DBF_WITH_CUDA -c $< -o $@

build/bevfusion_cuda: $(CUDA_C_OBJECTS) $(CUDA_RUNTIME_SOURCES)
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 $(CUDA_C_OBJECTS) $(CUDA_RUNTIME_SOURCES) -lcudnn -lcublas -o $@

build/test_cuda_runtime: tests/test_cuda_runtime.cu $(CUDA_RUNTIME_SOURCES) build/model_cuda_test.o build/frame_cuda_test.o include/bf_cuda_runtime.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_runtime.cu $(CUDA_RUNTIME_SOURCES) build/model_cuda_test.o build/frame_cuda_test.o -lcudnn -lcublas -o $@

build/test_cuda_lss: tests/test_cuda_lss.cu src/cuda_lss.cu build/model_cuda_test.o include/bf_cuda.h include/bf_lss.h include/bf_model.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_lss.cu src/cuda_lss.cu build/model_cuda_test.o -o $@

build/test_cuda_bev_stage: tests/test_cuda_bev_stage.cu src/cuda_bev_stage.cu build/model_cuda_test.o include/bf_cuda_bev.h include/bf_model.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_bev_stage.cu src/cuda_bev_stage.cu build/model_cuda_test.o -lcudnn -o $@

build/test_cuda_transfusion: tests/test_cuda_transfusion.cu src/cuda_transfusion.cu build/model_cuda_test.o include/bf_cuda_transfusion.h include/bf_transfusion_decoder.h include/bf_model.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_transfusion.cu src/cuda_transfusion.cu build/model_cuda_test.o -lcublas -o $@

build/test_cuda_tail: tests/test_cuda_tail.cu src/cuda_bev_stage.cu src/cuda_transfusion.cu build/model_cuda_test.o include/bf_cuda_bev.h include/bf_cuda_transfusion.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_tail.cu src/cuda_bev_stage.cu src/cuda_transfusion.cu build/model_cuda_test.o -lcudnn -lcublas -o $@

build/test_cuda_camera: tests/test_cuda_camera.cu src/cuda_camera.cu src/cuda_lss.cu build/model_cuda_test.o include/bf_cuda_camera.h include/bf_cuda.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_camera.cu src/cuda_camera.cu src/cuda_lss.cu build/model_cuda_test.o -lcudnn -o $@

build/test_cuda_swin: tests/test_cuda_swin.cu src/cuda_swin.cu build/model_cuda_test.o include/bf_cuda_swin.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_swin.cu src/cuda_swin.cu build/model_cuda_test.o -lcudnn -lcublas -o $@

build/test_cuda_camera_full: tests/test_cuda_camera_full.cu src/cuda_swin.cu src/cuda_camera.cu src/cuda_lss.cu build/model_cuda_test.o include/bf_cuda_swin.h include/bf_cuda_camera.h include/bf_cuda.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_camera_full.cu src/cuda_swin.cu src/cuda_camera.cu src/cuda_lss.cu build/model_cuda_test.o -lcudnn -lcublas -o $@

build/test_cuda_lidar: tests/test_cuda_lidar.cu src/cuda_lidar.cu src/cuda_voxel.cu build/model_cuda_test.o include/bf_cuda_lidar.h include/bf_cuda_voxel.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_lidar.cu src/cuda_lidar.cu src/cuda_voxel.cu build/model_cuda_test.o -o $@

build/test_cuda_voxel: tests/test_cuda_voxel.cu src/cuda_voxel.cu include/bf_cuda_voxel.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_voxel.cu src/cuda_voxel.cu -o $@

build/test_cuda_voxel_real: tests/test_cuda_voxel_real.cu src/cuda_voxel.cu build/voxel_cuda_test.o build/kernels_cuda_test.o build/frame_cuda_test.o include/bf_cuda_voxel.h
	mkdir -p build
	$(NVCC) $(CPPFLAGS) -O3 -arch=sm_89 tests/test_cuda_voxel_real.cu src/cuda_voxel.cu build/voxel_cuda_test.o build/kernels_cuda_test.o build/frame_cuda_test.o -lm -o $@

cuda-test: model build/kernel_oracle.bfw build/bev_stage_oracle.bfw build/transfusion_decoder_oracle.bfw build/cuda_tail_oracle.bfw build/image_fpn_oracle.bfw build/depth_head_oracle.bfw build/lss_downsample_oracle.bfw build/swin_backbone_oracle.bfw build/cuda_camera_full_oracle.bfw build/lidar_backbone_oracle.bfw build/test_cuda_depth_raster build/test_cuda_lss build/test_cuda_bev_stage build/test_cuda_transfusion build/test_cuda_tail build/test_cuda_camera build/test_cuda_swin build/test_cuda_camera_full build/test_cuda_voxel build/test_cuda_voxel_real build/test_cuda_lidar build/test_cuda_runtime build/bevfusion_cuda
	./build/test_cuda_depth_raster
	./build/test_cuda_lss build/kernel_oracle.bfw
	./build/test_cuda_bev_stage $(MODEL) build/bev_stage_oracle.bfw
	./build/test_cuda_transfusion $(MODEL) build/transfusion_decoder_oracle.bfw
	./build/test_cuda_tail $(MODEL) build/cuda_tail_oracle.bfw
	./build/test_cuda_camera $(MODEL) build/image_fpn_oracle.bfw build/depth_head_oracle.bfw build/lss_downsample_oracle.bfw
	./build/test_cuda_swin $(MODEL) build/swin_backbone_oracle.bfw
	./build/test_cuda_camera_full $(MODEL) build/cuda_camera_full_oracle.bfw
	./build/test_cuda_voxel
	./build/test_cuda_voxel_real runs/real-mini/frame0.bfi
	./build/test_cuda_lidar $(MODEL) build/lidar_backbone_oracle.bfw
	./build/test_cuda_runtime $(MODEL) runs/real-mini/frame0.bfi

test: model build/bevfusion build/test_model build/kernel_oracle.bfw build/depth_raster_oracle.bfw build/bev_stage_oracle.bfw build/swin_backbone_oracle.bfw build/image_fpn_oracle.bfw build/depth_head_oracle.bfw build/lss_downsample_oracle.bfw build/lidar_backbone_oracle.bfw build/transfusion_decoder_oracle.bfw build/test_kernels build/test_depth_raster build/test_lss build/test_swin build/test_transfusion build/test_bev_stage build/test_swin_backbone build/test_image_fpn build/test_depth_head build/test_lss_downsample build/test_lidar_backbone build/test_transfusion_decoder build/test_runtime build/test_tui build/test_frame
	./build/test_model $(MODEL)
	./build/test_kernels build/kernel_oracle.bfw
	./build/test_depth_raster build/depth_raster_oracle.bfw
	./build/test_lss build/kernel_oracle.bfw
	./build/test_swin build/kernel_oracle.bfw
	./build/test_transfusion build/kernel_oracle.bfw
	./build/test_bev_stage $(MODEL) build/bev_stage_oracle.bfw
	./build/test_swin_backbone $(MODEL) build/swin_backbone_oracle.bfw
	./build/test_image_fpn $(MODEL) build/image_fpn_oracle.bfw
	./build/test_depth_head $(MODEL) build/depth_head_oracle.bfw
	./build/test_lss_downsample $(MODEL) build/lss_downsample_oracle.bfw
	./build/test_lidar_backbone $(MODEL) build/lidar_backbone_oracle.bfw
	./build/test_transfusion_decoder $(MODEL) build/transfusion_decoder_oracle.bfw
	./build/test_runtime $(MODEL)
	./build/test_tui
	./build/test_frame
	./build/bevfusion inspect $(MODEL)
	./build/bevfusion plan $(MODEL) 300000 160000 160000

portable-test:
	$(MAKE) clean
	$(MAKE) CFLAGS='-O2 -std=c11 -Wall -Wextra -Wpedantic' test

clean:
	rm -rf build
