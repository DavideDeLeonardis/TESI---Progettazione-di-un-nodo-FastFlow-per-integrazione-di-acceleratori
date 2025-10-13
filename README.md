# Integrazione FPGA in FastFlow

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
L'eseguibile accetta tre argomenti opzionali: [N] [NUM_TASKS] [DEVICE].
```
# Esecuzione su CPU (parallela)
./build/tesi-exec 1000000 100 cpu

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
L'eseguibile accetta tre argomenti opzionali: [N] [NUM_TASKS] [DEVICE].
```
# Esecuzione su CPU (parallela)
./build/tesi-exec 16777216 100 cpu

# Esecuzione su GPU
./build/tesi-exec 16777216 100 gpu
```