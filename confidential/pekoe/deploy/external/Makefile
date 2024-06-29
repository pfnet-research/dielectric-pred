DIRS = \
	pytorch \
	libtorch-dev \
	pfvm \
	obfuscator

.PHONY: all
all:install

.PHONY: install
install:
	for d in $(DIRS); do \
		cd $${d} && make install && cd ..; \
	done

.PHONY: clean
clean:
	for d in $(DIRS); do \
		cd $${d} && make clean && cd ..; \
	done
