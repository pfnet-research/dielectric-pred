.PHONY: all
all: prod

.PHONY: prod
prod:
	make -C external all install
	make -C teanet prod

.PHONY: dev
dev:
	make -C external all install
	make -C teanet dev

.PHONY: clean
clean:
	make -C external clean
	make -C teanet clean
