#
# Makefile for building pfvm. This Makefile supports several options like following:
#
# Options:
#    F                     Build force (default: 0). Build files even when they exist.
#    BUILD_INTERMEDIATE    Build intermediates when there are not (default: 1).
#                          If 0 is specified and there is a target file, it does not build the intermediates
#    BUILD_PREFIX          Build prefix directory (default: $(PWD)/..).
#    BUILD_DIR             Build oupout directory (default: $(BUILD_PREFIX)/pfvm/build)
#    INSTALL_PREFIX        Install prefix directory (default: $(PWD)/../../install)
#
# Examples:
#    $ make
#    $ make F=1
#    $ make BUILD_DIR=build.tmp
#
PWD = $(shell pwd)
PROJECT_NAME = $(notdir $(PWD))
BUILD_PREFIX ?= $(PWD)/..
BUILD_DIR ?= $(BUILD_PREFIX)/$(PROJECT_NAME)/build
BUILD_DIR := $(abspath $(BUILD_DIR))
INSTALL_PREFIX ?= $(PWD)/../../install
DEPLOY_PYTHON_VERSION ?= 3.8

PFVM_COMMIT_ID = 9859fe2847a2b9087ebced7a53466295daa9fa14

F ?= 0
BUILD_INTERMEDIATE ?= 1

.PHONY: all
all: install

ifeq ($(BUILD_INTERMEDIATE),0)
.INTERMEDIATE: pfvm
.PRECIOUS: pfvm # .PRECIOUS prevent intermediate files to be deleted
endif
pfvm:
	git clone git@github.pfidev.jp:model-deployment/pfvm.git \
	&& cd pfvm \
	&& git checkout $(PFVM_COMMIT_ID) \
	&& git apply $(PWD)/pfvm.patch \
	&& git submodule update --init --recursive \
	&& cd third_party/nms \
	&& git apply $(PWD)/nms.patch

.PHONY: build
build: $(BUILD_DIR)/compiler/libpfvm_compiler.a
ifeq ($(F),1)
.PHONY: $(BUILD_DIR)/compiler/libpfvm_compiler.a
endif
ifeq ($(BUILD_INTERMEDIATE),0)
.INTERMEDIATE: $(BUILD_DIR)/compiler/libpfvm_compiler.a
.PRECIOUS: $(BUILD_DIR)/compiler/libpfvm_compiler.a
endif
$(BUILD_DIR)/compiler/libpfvm_compiler.a: $(BUILD_DIR)/build.ninja
	cd $(BUILD_DIR) && cmake . && cmake --build .
	touch $@

ifeq ($(F),1)
.PHONY: $(BUILD_DIR)/build.ninja
endif
ifeq ($(BUILD_INTERMEDIATE),0)
.INTERMEDIATE: $(BUILD_DIR)/build.ninja
.PRECIOUS: $(BUILD_DIR)/build.ninja
endif
$(BUILD_DIR)/build.ninja: pfvm
	mkdir -p $(BUILD_DIR) \
	&& cd $(BUILD_DIR) \
	&& cmake $(PWD)/pfvm \
	      -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
	      -DPFVM_TORCH_DIR=$(INSTALL_PREFIX)/libtorch-dev \
	      -DPFVM_ENABLE_CUDA=ON \
	      -DPFVM_ENABLE_CUDNN=ON \
	      -DPFVM_STRIP_LOG=ON \
	      -DPFVM_GENERATE_TESTS=OFF \
	      -DPFVM_ENABLE_TEST=OFF \
	      -DPFVM_BUILD_TESTS=OFF \
	      -DPFVM_ENABLE_NVRTC=OFF \
	      -DPFVM_ENABLE_NVTX=OFF \
	      -Dpfnms_ENABLE_TEST=OFF \
	      -DPYTHON_EXECUTABLE=$(shell which python$(DEPLOY_PYTHON_VERSION)) \
	      -DCMAKE_BUILD_TYPE=Debug \
	      -DCUDA_TOOLKIT_ROOT_DIR=$(shell pkg-config --variable=cudaroot cuda-10.1) \
	      -G Ninja

.PHONY: install
install: $(INSTALL_PREFIX)/lib/libpfvm_compiler.a
ifeq ($(F),1)
.PHONY: $(INSTALL_PREFIX)/lib/libpfvm_compiler.a
endif
$(INSTALL_PREFIX)/lib/libpfvm_compiler.a: $(BUILD_DIR)/compiler/libpfvm_compiler.a
	cd $(BUILD_DIR) \
	&& cmake --build . --target third_party/nms/third_party/cpu_features/install \
	&& cmake --build . --target third_party/nms/install \
	&& cmake --build . --target third_party/onnx/install \
	&& cmake --build . --target third_party/protobuf/install \
	&& cmake --build . --target install \
	&& touch $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
