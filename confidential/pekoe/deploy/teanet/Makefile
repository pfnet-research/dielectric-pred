#
# Makefile for building binary model. This Makefile supports several options like following:
#
# Options:
#    BUILD_PREFIX             Build prefix directory (default: $(PWD)/..).
#    BUILD_DIR                Build oupout directory (default: $(BUILD_PREFIX)/teanet/build)
#    DEPLOY_INSTALL_PREFIX    Deploy libraries installed prefix (default: $(PWD)/../install).
#
# Examples:
#    $ make
#    $ make BUILD_DIR=build.tmp
#    $ make DEPLOY_INSTALL_PREFIX=/opt/pfn/pfp/deploy
#
PWD = $(shell pwd)
PROJECT_NAME = $(notdir $(PWD))

BUILD_PREFIX ?= $(PWD)/..
BUILD_DIR ?= $(BUILD_PREFIX)/$(PROJECT_NAME)/build
BUILD_DIR := $(abspath $(BUILD_DIR))

INSTALL_DIR = $(PWD)/lib
INSTALL2BUILD = $(shell realpath --relative-to=$(INSTALL_DIR) $(BUILD_DIR))

DEPLOY_PYTHON_VERSION ?= 3.8
RODATA_SEED := 0xcf361b38
PYTHON := $(shell which python$(DEPLOY_PYTHON_VERSION))

DEPLOY_INSTALL_PREFIX ?= $(PWD)/../install

TORCH_INSTALL_PREFIX = $(DEPLOY_INSTALL_PREFIX)/libtorch
TORCH_DEV_INSTALL_PREFIX = $(DEPLOY_INSTALL_PREFIX)/libtorch-dev
PFVM_INSTALL_PREFIX = $(DEPLOY_INSTALL_PREFIX)
OBFUSCATOR_INSTALL_PREFIX = $(DEPLOY_INSTALL_PREFIX)

RUNTIME_LIBS := \
	-Wl,--whole-archive $(BUILD_DIR)/libinfer_static.a -Wl,--no-whole-archive \
	$(PFVM_INSTALL_PREFIX)/lib/libpfvm_runtime_static.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfvm_common.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms_sort_bmi1.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms_sort_generic.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms_loop_avx.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms_loop_avx512.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms_loop_generic.a \
	$(PFVM_INSTALL_PREFIX)/lib/libpfnms_runtime.a \
	$(PFVM_INSTALL_PREFIX)/lib/libcpu_features.a \
	-L$(TORCH_INSTALL_PREFIX)/lib \
	-Wl,--whole-archive \
	-lc10_cuda \
	-ltorch \
	-ltorch_cpu \
	-ltorch_cuda \
	-lXNNPACK \
	-Wl,--no-whole-archive \
	-L$(TORCH_DEV_INSTALL_PREFIX)/lib \
	-lc10 \
	-lfoxi_loader \
	-lcaffe2_protos \
	-lCaffe2_perfkernels_avx \
	-lCaffe2_perfkernels_avx2 \
	-lonnx \
	-lonnx_proto \
	-lcpuinfo \
	-lclog \
	-lsleef \
	-L/usr/local/cuda/lib64 \
	-lcudnn_static \
	-lcudart_static \
	-lcusolver_static \
	-lcusparse_static \
	-llapack_static \
	-lcublas_static \
	-lcublasLt_static \
	-lcufft_static_nocallback \
	-lcufft_static \
	-lculibos \
	-lcurand_static \
	-lnvToolsExt \
	-lopenblas \
	-lrt \
	-ldl \
	$(PFVM_INSTALL_PREFIX)/lib/libprotobufd.a \
	-Wl,--version-script,$(OBFUSCATOR_INSTALL_PREFIX)/lib/obfuscator/minimal_pybind.map \
	-Wl,-T,$(OBFUSCATOR_INSTALL_PREFIX)/lib/obfuscator/mark_rodata.lds \


all: prod

prod: $(BUILD_DIR)/libinfer_prod.so
	ln -sf $(INSTALL2BUILD)/libinfer_prod.so $(INSTALL_DIR)/libinfer.so

dev: $(BUILD_DIR)/libinfer_dev.so
	ln -sf $(INSTALL2BUILD)/libinfer_dev.so $(INSTALL_DIR)/libinfer.so

build-prod: $(BUILD_DIR)/libinfer_prod.so
$(BUILD_DIR)/libinfer_prod.so: $(BUILD_DIR)/build.ninja Makefile
	ninja -C $(BUILD_DIR) infer_static
	$(CXX) -shared $(RUNTIME_LIBS) -o $(BUILD_DIR)/libinfer.so
	cp $(BUILD_DIR)/libinfer.so $(BUILD_DIR)/libinfer_prod.so
	$(PYTHON) $(OBFUSCATOR_INSTALL_PREFIX)/lib/obfuscator/encode_rodata.py --seed $(RODATA_SEED) --binary $(BUILD_DIR)/libinfer_prod.so --encode_bin $(OBFUSCATOR_INSTALL_PREFIX)/bin/encode

build-dev: $(BUILD_DIR)/libinfer_dev.so
$(BUILD_DIR)/libinfer_dev.so: $(BUILD_DIR)/build.ninja
	ninja -C $(BUILD_DIR) libinfer_dev.so

$(BUILD_DIR)/build.ninja: target_onnx/model.onnx Makefile CMakeLists.txt
	mkdir -p $(BUILD_DIR) \
	&& cd $(BUILD_DIR) \
	&& cmake $(PWD) \
	-DTORCH_DEV_INSTALL_PREFIX=$(TORCH_DEV_INSTALL_PREFIX) \
	-DPFVM_INSTALL_PREFIX=$(PFVM_INSTALL_PREFIX) \
	-DOBFUSCATOR_INSTALL_PREFIX=$(OBFUSCATOR_INSTALL_PREFIX) \
	-DPYTHON_EXECUTABLE=$(PYTHON) \
	-DCMAKE_BUILD_TYPE=Debug \
	-DCUDA_TOOLKIT_ROOT_DIR=$(shell pkg-config --variable=cudaroot cuda-10.1) \
	-G Ninja

clean:
	rm -rf $(BUILD_DIR) $(INSTALL_DIR)/libinfer*.so

.PHONY: all build-prod build-dev $(BUILD_DIR)/libinfer_dev.so $(BUILD_DIR)/libinfer_prod.so clean

-include *.d
