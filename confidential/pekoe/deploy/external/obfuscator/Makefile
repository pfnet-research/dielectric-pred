#
# Makefile for building obfuscator. This Makefile supports several options like following:
#
# Options:
#    F                     Build force (default: 0). Build files even when they exist.
#    BUILD_INTERMEDIATE    Build intermediates when there are not (default: 1).
#                          If 0 is specified and there is a target file, it does not build the intermediates
#    BUILD_PREFIX          Build prefix directory (default: $(PWD)/..).
#    BUILD_DIR             Build oupout directory (default: $(BUILD_PREFIX)/obfuscator/build)
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

OBFUSCATOR_COMMIT_ID = e174393904324ba4a5222c066cb092854d0635b8

F ?= 0
BUILD_INTERMEDIATE ?= 1

.PHONY: all
all: install

ifeq ($(BUILD_INTERMEDIATE),0)
.INTERMEDIATE: obfuscator
.PRECIOUS: obfuscator
endif
obfuscator:
	git clone git@github.pfidev.jp:model-deployment/obfuscator.git \
	&& cd obfuscator \
	&& git checkout $(OBFUSCATOR_COMMIT_ID) \
	&& git apply $(PWD)/obfuscator.patch

.PHONY: build
build: $(BUILD_DIR)/encode
ifeq ($(F),1)
.PHONY: $(BUILD_DIR)/encode
endif
ifeq ($(BUILD_INTERMEDIATE),0)
.INTERMEDIATE: $(BUILD_DIR)/encode
.PRECIOUS: $(BUILD_DIR)/encode
endif
$(BUILD_DIR)/encode: $(BUILD_DIR)/build.ninja
	cd $(BUILD_DIR) && cmake . && cmake --build .
	touch $@

ifeq ($(F),1)
.PHONY: $(BUILD_DIR)/build.ninja
endif
ifeq ($(BUILD_INTERMEDIATE),0)
.INTERMEDIATE: $(BUILD_DIR)/build.ninja
.PRECIOUS: $(BUILD_DIR)/build.ninja
endif
$(BUILD_DIR)/build.ninja: obfuscator
	mkdir -p $(BUILD_DIR) \
	&& cd $(BUILD_DIR) \
	&& cmake $(PWD)/obfuscator \
	      -DCMAKE_BUILD_TYPE=Debug \
	      -DCMAKE_CXX_STANDARD=14 \
	      -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX) \
	      -G Ninja

.PHONY: install
install: $(INSTALL_PREFIX)/bin/encode
ifeq ($(F),1)
.PHONY: $(INSTALL_PREFIX)/bin/encode
endif
$(INSTALL_PREFIX)/bin/encode: $(BUILD_DIR)/encode
	cd $(BUILD_DIR) \
	&& cmake --build . --target install \
	&& touch $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
