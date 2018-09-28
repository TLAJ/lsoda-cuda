CC       = nvcc -O2 -arch=sm_61 -maxrregcount=32 #-Xptxas -dlcm=ca #--fmad=false
CFLAGS   = -shared -Xcompiler -fPIC
LDFLAGS  = 

SOS = sample.so

all: ${SOS}

%.so: %.cu
	${CC} ${CFLAGS} ${LDFLAGS} -o $@ $<

clean:
	rm -f ${SOS}

# -----------------------------------------------------------------------------------------

