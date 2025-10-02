#pragma once
#include "../include/types.hpp"
#include "fastflow_includes.hpp"
#include <atomic>
#include <iostream>
#include <thread>

// Includi l'header di OpenCL. La guardia #ifdef è per la portabilità
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/* Nodo accelerato con doppia coda e due thread */
class ff_node_acc_t : public ff_node {
 public:
   ff_node_acc_t();
   ~ff_node_acc_t(); // Useremo il distruttore per la pulizia

   long long getComputeTime_us() const;

   int svc_init() override;
   void *svc(void *t) override;
   void svc_end() override;

 private:
   static void *const SENTINEL;
   void producerLoop();
   void consumerLoop();

   // --- Membri per OpenCL ---
   cl_context context_{nullptr};
   cl_command_queue queue_{nullptr};
   cl_program program_{nullptr};
   cl_kernel kernel_{nullptr};

   using TaskQ = uSWSR_Ptr_Buffer;
   using ResultQ = uSWSR_Ptr_Buffer;

   TaskQ *inQ_{nullptr};
   ResultQ *outQ_{nullptr};

   std::thread prodTh_, consTh_;
   std::atomic<long long> computed_us_{0};
   std::atomic<size_t> inPushed_, inPopped_;
   std::atomic<size_t> outPushed_, outPopped_;
};