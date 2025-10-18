# Progettazione di un nodo FastFlow per integrazione di acceleratori

<br>
Ci sono 3 kernel disponibili per GPU e FPGA, per CPU è disponibile solo l'operazione polinomiale, che è l'operazione inserita di default su ogni dispositivo.
<br>
Il testing è fatto per CPU con FF, CPU con OpenMP, GPU con OpenCL, GPU con Metal e FPGA con il kernel polynomial_op.
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
