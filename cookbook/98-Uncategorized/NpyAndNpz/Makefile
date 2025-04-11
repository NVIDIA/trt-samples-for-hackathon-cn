include $(TRT_COOKBOOK_PATH)/include/Makefile.inc

all: main.exe

main.exe: main.o cookbookHelper.o
	$(COMMAND_EXE) -lz

main.o: main.cpp
	$(COMMAND_O)

.PHONY: test
test:
	python3 get_data.py
	make all -j
	./main.exe > log-main.exe.log

.PHONY: clean
clean:
	rm -rf *.d *.o *.exe *.log *.npy *.npz
