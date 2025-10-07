#include "GpuAccelerator.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

GpuAccelerator::GpuAccelerator() {
   std::cerr << "[INFO] GpuAccelerator: Created.\n";
}

/**
 * @brief Distruttore della classe GpuAccelerator.
 *
 * Si occupa di rilasciare in modo sicuro tutte le risorse OpenCL allocate
 * durante il ciclo di vita dell'oggetto. Le risorse vengono rilasciate in
 * ordine inverso rispetto alla loro creazione.
 */
GpuAccelerator::~GpuAccelerator() {
   std::cerr << "[INFO] GpuAccelerator: Cleaning up OpenCL resources...\n";
   if (kernel_)
      clReleaseKernel(kernel_);
   if (program_)
      clReleaseProgram(program_);
   if (queue_)
      clReleaseCommandQueue(queue_);
   if (context_)
      clReleaseContext(context_);
   std::cerr << "[INFO] GpuAccelerator: Destroyed.\n";
}

/**
 * @brief Inizializza l'ambiente OpenCL per l'uso della GPU.
 *
 * Esegue tutte le operazioni di setup una tantum: trova il dispositivo,
 * crea il contesto e la coda di comandi, legge il sorgente del kernel,
 * lo compila e prepara l'oggetto kernel per l'esecuzione.
 */
bool GpuAccelerator::initialize() {
   std::cerr << "[GpuAccelerator] Initializing...\n";
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;
   cl_int ret;

   // Trova una piattaforma OpenCL e un dispositivo di tipo GPU.
   clGetPlatformIDs(1, &platform_id, NULL);
   clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

   // Crea un contesto OpenCL.
   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create OpenCL context.\n";
      return false;
   }

   // Crea una coda di comandi. È il canale attraverso cui l'host
   // invia le operazioni (es. trasferimenti di memoria, esecuzioni kernel)
   // al device.
   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create command queue.\n";
      return false;
   }

   // Legge il codice sorgente del kernel OpenCL da un file esterno.
   std::ifstream kernelFile("kernel/vecAdd.cl");
   if (!kernelFile.is_open()) {
      std::cerr
         << "[ERROR] GpuAccelerator: Could not open kernel file vecAdd.cl\n";
      return false;
   }
   std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)),
                            (std::istreambuf_iterator<char>()));
   const char *source_str = kernelSource.c_str();
   size_t source_size = kernelSource.length();

   // Crea un oggetto programma OpenCL a partire dal codice sorgente.
   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create program.\n";
      return false;
   }

   // Compila il programma per il dispositivo target.
   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Kernel compilation failed.\n";
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      std::cerr << "--- KERNEL BUILD LOG ---\n" << log.data() << "\n";
      return false;
   }

   // Estrai un handle al kernel compilato ("vecAdd").
   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "[ERROR] GpuAccelerator: Failed to create kernel object.\n";
      return false;
   }

   std::cerr << "[GpuAccelerator] Initialization successful.\n";
   return true;
}

/**
 * @brief Esegue un singolo task di somma vettoriale sulla GPU.
 *
 * Orchestra l'intero processo di offloading: allocazione della memoria sul
 * device, trasferimento dei dati, impostazione degli argomenti, esecuzione
 * del kernel e recupero dei risultati.
 * @param generic_task Puntatore generico al task da eseguire.
 * @param computed_us Riferimento per restituire il tempo di calcolo.
 */
void GpuAccelerator::execute(void *generic_task, long long &computed_us) {
   auto *task = static_cast<Task *>(generic_task);
   std::cerr << "[GpuAccelerator] Offloading task with N=" << task->n
             << "...\n";

   cl_int ret;
   size_t buffer_size = sizeof(int) * task->n;

   // Allocazione dei buffer di memoria sulla GPU (device memory).
   // CL_MEM_READ_ONLY: l'host può solo scriverci, il kernel solo leggerci.
   // CL_MEM_WRITE_ONLY: il kernel può solo scriverci, l'host solo leggerci.
   cl_mem bufferA =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferB =
      clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
   cl_mem bufferC =
      clCreateBuffer(context_, CL_MEM_WRITE_ONLY, buffer_size, NULL, &ret);

   auto t0 = std::chrono::steady_clock::now();

   // Trasferimento dei dati dalla memoria host (RAM) alla memoria device
   // (VRAM). CL_TRUE: la chiamata è bloccante: attende il completamento del
   // trasferimento.
   clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0, buffer_size, task->a, 0,
                        NULL, NULL);
   clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0, buffer_size, task->b, 0,
                        NULL, NULL);

   // Impostazione degli argomenti del kernel. Si collegano i buffer di
   // memoria del device ai parametri della funzione kernel.
   clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA);
   clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB);
   clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC);
   clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n));

   // Esecuzione del kernel.
   // Si specifica il numero totale di thread (work-item) da lanciare,
   // che in questo caso è pari alla dimensione del vettore.
   size_t global_work_size = task->n;
   clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size, NULL, 0,
                          NULL, NULL);

   // Trasferimento dei risultati dalla memoria device alla memoria host.
   clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, buffer_size, task->c, 0,
                       NULL, NULL);

   // Sincronizzazione. Forza l'attesa del completamento di tutti i comandi
   // inviati alla coda.
   clFinish(queue_);

   auto t1 = std::chrono::steady_clock::now();
   computed_us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

   // Rilascio dei buffer di memoria sul device, libera le risorse.
   clReleaseMemObject(bufferA);
   clReleaseMemObject(bufferB);
   clReleaseMemObject(bufferC);
}