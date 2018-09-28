CC       = g++
CFLAGS   = -shared -fPIC
LDFLAGS  = 

SOS = sample.so

all: ${SOS}

%.so: %.cpp
	${CC} ${CFLAGS} ${LDFLAGS} -o $@ $<

clean:
	rm -f ${SOS}

# -----------------------------------------------------------------------------------------

