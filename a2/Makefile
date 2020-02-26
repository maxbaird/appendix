PROJECT1  = reduction_native.out
PROJECT2  = reduction_interrupt.out
CC        = nvcc
CFLAGS    = -c -Xcompiler -Wall,-Werror,-Wextra

.PHONY : all clean

all : $(PROJECT1) $(PROJECT2)

$(PROJECT1) : reduction_native.o
	$(CC) $< -o $@

$(PROJECT2) : reduction_interrupt.o
	$(CC) $< -o $@

reduction_native.o : reduction_native.cu
	$(CC) $(CFLAGS) $< -o $@

reduction_interrupt.o : reduction_interrupt.cu
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(PROJECT1) $(PROJECT2)
