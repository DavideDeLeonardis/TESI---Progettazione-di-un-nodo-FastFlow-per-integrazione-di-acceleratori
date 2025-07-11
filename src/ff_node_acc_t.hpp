#pragma once
#include "../include/types.hpp"
#include <ff/node.hpp>

class ff_node_acc_t : public ff::ff_node {
 public:
   ff_node_acc_t() = default;
   ~ff_node_acc_t() = default;

   // FastFlow lifecycle
   int svc_init() override;        // (vuoto per CPU)
   void *svc(void *task) override; // somma vettoriale
   void svc_end() override;        // (vuoto per CPU)
};
