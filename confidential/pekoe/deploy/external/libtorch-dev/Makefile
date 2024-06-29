PWD = $(shell pwd)
PROJECT_NAME = $(notdir $(PWD))
BUILD_PREFIX ?= $(PWD)/..
BUILD_DIR ?= $(BUILD_PREFIX)/$(PROJECT_NAME)/build
BUILD_DIR := $(abspath $(BUILD_DIR))
INSTALL_PREFIX ?= $(PWD)/../../install
TORCH_VERSION ?= 1.5.0

F ?= 0

.PHONY: all
all: install

.PHONY: zip
zip: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip
ifeq ($(F),1)
.PHONY: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip
endif
.INTERMEDIATE: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip
.PRECIOUS: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip
$(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip:
	mkdir -p $(BUILD_DIR)
	curl -L https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-$(TORCH_VERSION).zip > $@

.PHONY: install
install: $(INSTALL_PREFIX)/libtorch-dev/lib/libfoxi_loader.a
ifeq ($(F),1)
.PHONY: $(INSTALL_PREFIX)/libtorch-dev/lib/libfoxi_loader.a
endif
$(INSTALL_PREFIX)/libtorch-dev/lib/libfoxi_loader.a: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip
	unzip -q -d $(BUILD_DIR) $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-dev.zip
	mv $(BUILD_DIR)/libtorch $(BUILD_DIR)/libtorch-dev
	mkdir -p $(INSTALL_PREFIX)
	tar c -C $(BUILD_DIR) libtorch-dev | tar x -C $(INSTALL_PREFIX)
	rm -rf $(BUILD_DIR)/libtorch-dev
	touch $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
