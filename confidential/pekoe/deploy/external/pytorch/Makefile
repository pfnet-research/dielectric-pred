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
zip: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-cu101.zip
ifeq ($(F),1)
.PHONY: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-cu101.zip
endif
.INTERMEDIATE: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-cu101.zip
.PRECIOUS: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-cu101.zip
$(BUILD_DIR)/libtorch-$(TORCH_VERSION)-cu101.zip:
	mkdir -p $(BUILD_DIR)
ifeq ($(shell grep -e docker -e containerd /proc/1/cgroup),)
	$(eval CONTAINER := $(shell docker create asia-northeast1-docker.pkg.dev/pfn-artifactregistry/chem/pekoe-libtorch-binary))
	docker cp $(CONTAINER):/libtorch-$(TORCH_VERSION)-cu101.zip $@
	docker rm $(CONTAINER)
else
	cp /tmp/pfp/pytorch/libtorch-${TORCH_VERSION}-cu101.zip $@
endif
	touch $@

.PHONY: install
install: $(INSTALL_PREFIX)/libtorch/lib/libtorch.a
ifeq ($(F),1)
.PHONY: $(INSTALL_PREFIX)/libtorch/lib/libtorch.a
endif
$(INSTALL_PREFIX)/libtorch/lib/libtorch.a: $(BUILD_DIR)/libtorch-$(TORCH_VERSION)-cu101.zip
	mkdir -p $(INSTALL_PREFIX)
	unzip -o -d $(INSTALL_PREFIX) $<
	touch $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
