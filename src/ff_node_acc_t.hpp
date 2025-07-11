#pragma once
#include <ff/buffer.hpp>
#include <ff/node.hpp>
#include <thread>
#include "../include/types.hpp"

// class ff_node_acc_t : public ff::ff_node {
//  public:
//    ff_node_acc_t() = default;
//    ~ff_node_acc_t() noexcept override = default;

//    int svc_init() override;
//    void *svc(void *t) override;
//    void svc_end() override;

//  private:
//    ff::SWSR_Ptr_Buffer *inQ_{nullptr};
//    std::thread prodTh_;

//    void producerLoop();
// };

class ff_node_acc_t : public ff::ff_node {
 public:
   void *svc(void *t) override;
};
