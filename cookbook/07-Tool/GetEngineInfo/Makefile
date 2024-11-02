include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

all: main.exe

main.exe: main.o cookbookHelper.o
	$(COMMAND_EXE)

main.o: main.cpp
	$(COMMAND_O)

.PHONY: test
test:
	make clean; make all -j
	./main.exe model.trt > log-main.exe.log

.PHONY: clean
clean:
	rm -rf *.d *.o *.exe
