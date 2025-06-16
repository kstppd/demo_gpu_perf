PLATFORM ?= nvidia
ifeq ($(PLATFORM), nvidia)
	COMPILER   = nvcc
	CXXFLAGS   = -std=c++20 -O3 --expt-relaxed-constexpr
	XCOMPFLAGS = -Wall -Wextra
	LIBS       = -lnvToolsExt -lcublas -lblas
	TEST2_INCLUDES = -I./external/asterix/include -isystem=./external/spdlog/include/ -system=./external/stb/ 
else ifeq ($(PLATFORM), amd)
	COMPILER   = hipcc
	CXXFLAGS   = -std=c++20 -O3 -DUSE_HIP -DSKIP_HOSTBLAS
	XCOMPFLAGS = -Wall -Wextra
	INCLUDES  += -I/opt/rocm/roctracer/include -I/opt/rocm/hipblas/include
	LDFLAGS    = -L/opt/rocm/roctracer/lib -L/opt/rocm/hipblas/lib	
	LIBS       = -lroctx64 -lhipblas   
	TEST2_INCLUDES = -I./external/asterix/include -I./external/spdlog/include/ -I./external/stb/ 
else
	$(error PLATFORM must be either 'nvidia' or 'amd')
endif
TEST2_LD =  -L./external/spdlog/build -lspdlog  

deps:
	if [ ! -d "external/asterix" ]; then mkdir -p external ; git clone  --depth 1 https://github.com/kstppd/asterix.git external/asterix/; fi
	if [ ! -d "external/spdlog" ]; then  git clone --depth 1  https://github.com/gabime/spdlog.git external/spdlog/; fi
	if [ ! -d "external/stb" ]; then  git clone --depth 1  https://github.com/nothings/stb.git external/stb/; fi
	if [ ! -d "external/spdlog/build/" ]; then cd external/spdlog && mkdir -p build && cd build && cmake .. && cmake --build . -j=8; fi

all: deps test1 test2

test1: deps
	$(COMPILER) -DDRY $(CXXFLAGS) test1/main.cu -Xcompiler "$(XCOMPFLAGS)" $(INCLUDES) $(SRC) -o bench1 $(LDFLAGS) $(LIBS)
	
test2: deps
	$(COMPILER) $(CXXFLAGS) $(TEST2_INCLUDES) $(TEST2_LD) test2/image.cu -Xcompiler "$(XCOMPFLAGS)" $(INCLUDES) $(SRC) -o bench2 $(LDFLAGS) $(LIBS)
	
test3: deps
	$(COMPILER) -DDRY $(CXXFLAGS) test3/eulernv.cu -Xcompiler "$(XCOMPFLAGS)" $(INCLUDES) $(SRC) -o bench3 $(LDFLAGS) $(LIBS)	
	
clean:
	rm bench1
	rm bench2
