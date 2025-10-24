# Progettazione di un nodo FastFlow per integrazione di acceleratori


Sono disponibili 4 kernel per GPU e FPGA, 3 per CPU.
<br>

## Su VM Pianosa

Prima di compilare o eseguire, attivare l'ambiente Vitis nella directory principale della VM:

```
source my_settings64.sh
```

### Compilazione

```
cd Tesi;
rm -rf build;
cmake -B build && cmake --build build
```

### Esecuzione

L'eseguibile accetta tre argomenti opzionali: [N] [NUM_TASKS] [DEVICE] [KERNEL_PATH].

```
# Esecuzione su CPU (con ff::parallel_for)
./build/tesi-exec 1000000 100 cpu_ff

# Esecuzione su CPU (con OpenMP)
./build/tesi-exec 1000000 100 cpu_omp

# Esecuzione su FPGA
./build/tesi-exec 1000000 100 fpga
```

<br>

## Su MacOS

### Compilazione

```
cd Tesi;
rm -rf build;
cmake -B build && cmake --build build
```

### Esecuzione

L'eseguibile accetta tre argomenti opzionali: [N] [NUM_TASKS] [DEVICE] [KERNEL_PATH].

```
# Esecuzione su CPU (con ff::parallel_for)
./build/tesi-exec 16777216 100 cpu_ff

# Esecuzione su GPU (OpenCL)
./build/tesi-exec 16777216 100 gpu_opencl

# Esecuzione su GPU (Metal)
./build/tesi-exec 16777216 100 gpu_metal
```
