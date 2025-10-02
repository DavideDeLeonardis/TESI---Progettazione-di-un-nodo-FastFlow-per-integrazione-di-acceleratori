#include "ff_node_acc_t.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

// Header C++ per leggere il file del kernel
#include <fstream>
#include <sstream>
#include <vector>

static char sentinel_obj;
void *const ff_node_acc_t::SENTINEL = &sentinel_obj;

ff_node_acc_t::ff_node_acc_t()
    : inPushed_(0), inPopped_(0), outPushed_(0), outPopped_(0) {}

// Il distruttore si occuperà della pulizia delle risorse OpenCL
ff_node_acc_t::~ff_node_acc_t() {
   std::cerr << "[destructor] Cleaning up OpenCL resources...\n";
   if (kernel_)
      clReleaseKernel(kernel_);
   if (program_)
      clReleaseProgram(program_);
   if (queue_)
      clReleaseCommandQueue(queue_);
   if (context_)
      clReleaseContext(context_);
}

long long ff_node_acc_t::getComputeTime_us() const {
   return computed_us_.load();
}

int ff_node_acc_t::svc_init() {
   // --- SETUP OPENCL ---
   cl_int ret;
   cl_platform_id platform_id = NULL;
   cl_device_id device_id = NULL;
   cl_uint ret_num_devices;
   cl_uint ret_num_platforms;

   ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to get Platform IDs\n";
      return -1;
   }
   ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                        &ret_num_devices);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to get Device IDs\n";
      return -1;
   }

   context_ = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
   if (!context_) {
      std::cerr << "Error: Failed to create OpenCL context\n";
      return -1;
   }

   queue_ = clCreateCommandQueue(context_, device_id, 0, &ret);
   if (!queue_) {
      std::cerr << "Error: Failed to create command queue\n";
      return -1;
   }

   // Carica il kernel dal file
   std::ifstream kernelFile("vecAdd.cl");
   if (!kernelFile.is_open()) {
      std::cerr << "Error: Could not open kernel file vecAdd.cl\n";
      return -1;
   }
   std::stringstream ss;
   ss << kernelFile.rdbuf();
   std::string kernelSource = ss.str();
   const char *source_str = kernelSource.c_str();
   size_t source_size = kernelSource.length();

   program_ =
      clCreateProgramWithSource(context_, 1, &source_str, &source_size, &ret);
   if (!program_) {
      std::cerr << "Error: Failed to create program\n";
      return -1;
   }

   ret = clBuildProgram(program_, 1, &device_id, NULL, NULL, NULL);
   if (ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to build program\n";
      // Stampa i log di compilazione per il debug
      size_t log_size;
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &log_size);
      std::vector<char> log(log_size);
      clGetProgramBuildInfo(program_, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), NULL);
      std::cerr << std::string(log.begin(), log.end()) << std::endl;
      return -1;
   }

   kernel_ = clCreateKernel(program_, "vecAdd", &ret);
   if (!kernel_ || ret != CL_SUCCESS) {
      std::cerr << "Error: Failed to create kernel\n";
      return -1;
   }

   std::cerr << "[svc_init] OpenCL setup completed successfully.\n";
   // --- FINE SETUP OPENCL ---

   inQ_ = new TaskQ(1024);
   outQ_ = new ResultQ(1024);
   if (!inQ_->init() || !outQ_->init()) {
      std::cerr << "[svc_init] ERROR: init queues failed\n";
      return -1;
   }
   prodTh_ = std::thread(&ff_node_acc_t::producerLoop, this);
   consTh_ = std::thread(&ff_node_acc_t::consumerLoop, this);
   std::cerr << "[svc_init] Threads started\n";
   return 0;
}

void *ff_node_acc_t::svc(void *t) {
   // Questa funzione rimane identica
   if (t == FF_EOS) {
      std::cerr << "[svc] received FF_EOS → pushing SENTINEL to inQ_\n";
      while (!inQ_->push(SENTINEL))
         std::this_thread::yield();
      return FF_GO_ON;
   }
   auto *task = static_cast<Task *>(t);
   std::cerr << "[svc] received TASK ptr=" << task << " n=" << task->n << "\n";
   while (!inQ_->push(task)) {
      std::cerr << "[svc] inQ_ full, retrying\n";
      std::this_thread::yield();
   }
   std::cerr << "[svc] task pushed to inQ_, returning FF_GO_ON\n";
   return FF_GO_ON;
}

void ff_node_acc_t::producerLoop() {
   void *ptr = nullptr;
   while (true) {
      // ... pop dalla coda inQ_ ... (codice invariato)
      std::cerr << "[producer] waiting for inQ_ pop...\n";
      while (!inQ_->pop(&ptr)) {
         std::this_thread::yield();
      }

      if (ptr == SENTINEL) {
         // ... gestione sentinella ... (codice invariato)
         while (!outQ_->push(SENTINEL)) {
            std::this_thread::yield();
         }
         break;
      }

      auto *task = static_cast<Task *>(ptr);
      std::cerr << "[producer] offloading task n=" << task->n << " to GPU\n";
      cl_int ret;

      // --- INIZIO LOGICA DI OFFLOADING PER OGNI TASK ---

      // 1. Crea i buffer di memoria sulla GPU
      size_t buffer_size = sizeof(int) * task->n;
      cl_mem bufferA =
         clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
      cl_mem bufferB =
         clCreateBuffer(context_, CL_MEM_READ_ONLY, buffer_size, NULL, &ret);
      cl_mem bufferC =
         clCreateBuffer(context_, CL_MEM_WRITE_ONLY, buffer_size, NULL, &ret);

      // --- INIZIO MISURAZIONE TEMPO ---
      auto t0 = std::chrono::steady_clock::now();

      // 2. Trasferisci i dati dalla CPU alla GPU (bloccante)
      clEnqueueWriteBuffer(queue_, bufferA, CL_TRUE, 0, buffer_size, task->a, 0,
                           NULL, NULL);
      clEnqueueWriteBuffer(queue_, bufferB, CL_TRUE, 0, buffer_size, task->b, 0,
                           NULL, NULL);

      // 3. Imposta gli argomenti del kernel
      clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufferA);
      clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferB);
      clSetKernelArg(kernel_, 2, sizeof(cl_mem), &bufferC);
      clSetKernelArg(kernel_, 3, sizeof(unsigned int), &(task->n));

      // 4. Esegui il kernel
      size_t global_work_size = task->n;
      clEnqueueNDRangeKernel(queue_, kernel_, 1, NULL, &global_work_size, NULL,
                             0, NULL, NULL);

      // 5. Recupera i risultati dalla GPU alla CPU (bloccante)
      clEnqueueReadBuffer(queue_, bufferC, CL_TRUE, 0, buffer_size, task->c, 0,
                          NULL, NULL);

      // Assicura che tutti i comandi siano finiti (utile per la misurazione)
      clFinish(queue_);

      auto t1 = std::chrono::steady_clock::now();
      auto us =
         std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      computed_us_ += us;
      // --- FINE MISURAZIONE TEMPO ---

      // 6. Rilascia i buffer creati per questo task
      clReleaseMemObject(bufferA);
      clReleaseMemObject(bufferB);
      clReleaseMemObject(bufferC);

      // --- FINE LOGICA DI OFFLOADING ---

      auto *res = new Result{task->c, task->n};
      std::cerr << "[producer] result ready\n";
      while (!outQ_->push(res)) {
         std::cerr << "[producer] outQ_ full, retrying push res\n";
         std::this_thread::yield();
      }
   }
   std::cerr << "[producer] exiting\n";
}

void ff_node_acc_t::consumerLoop() {
   void *ptr = nullptr;
   while (true) {
      std::cerr << "[consumer] waiting for outQ_ pop...\n";
      while (!outQ_->pop(&ptr))
         std::this_thread::yield();
      size_t opc = ++outPopped_;
      std::cerr << "[consumer] popped ptr (outPopped=" << opc << ")\n";

      if (ptr == SENTINEL) { // Controlla SENTINEL
         std::cerr << "[consumer] got SENTINEL → sending FF_EOS downstream\n";
         ff_send_out(FF_EOS);
         break;
      }

      std::cerr << "[consumer] sending result downstream\n";
      ff_send_out(ptr);
   }
   std::cerr << "[consumer] exiting\n";
}

void ff_node_acc_t::svc_end() {
   std::cerr
      << "[svc_end] Pipeline is shutting down. Signalling internal threads.\n";

   inQ_->push(SENTINEL); // Usa SENTINEL

   std::cerr << "[svc_end] joining producer thread\n";
   if (prodTh_.joinable())
      prodTh_.join();
   std::cerr << "[svc_end] producer thread joined\n";

   std::cerr << "[svc_end] joining consumer thread\n";
   if (consTh_.joinable())
      consTh_.join();
   std::cerr << "[svc_end] consumer thread joined\n";

   std::cerr << "[svc_end] Cleaning up resources.\n";
   delete inQ_;
   delete outQ_;
   inQ_ = nullptr;
   outQ_ = nullptr;
}