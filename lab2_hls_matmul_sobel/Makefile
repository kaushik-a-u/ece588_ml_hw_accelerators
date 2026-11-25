VHLS_PATH := $(dir $(shell which vitis_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

CFLAGS = -g -I${VHLS_INC} -std=c++11

csim: sobel.cpp sobel_test.cpp
	@echo "Simulating the sobel Program"
	g++ ${CFLAGS} $^ -o sobel -lrt
	./sobel

csyn:	
	@echo "Running VitisHLS by simulating and compiling the Program"
	rm -rf sobel
	vitis_hls -f tcl_script.tcl

clean:
	rm -rf sobel *.dat *prj *.log
