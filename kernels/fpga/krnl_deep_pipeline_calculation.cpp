/*******************************************************************************
Description:

    This kernel demonstrates a deep pipeline with 4 stages, implemented
    using the load/compute/store style for Vitis HLS. Each stage is a
    separate function connected by HLS streams. The #pragma HLS dataflow
    enables task-level pipelining, allowing all stages to operate in parallel.

*******************************************************************************/

// Includes
#include <hls_stream.h>
#include <stdint.h>

#define DATA_SIZE 4096
const int c_size = DATA_SIZE;

// --- Pipeline Stages ---

static void stage1(hls::stream<int32_t> &in_a, hls::stream<int32_t> &in_b,
                   hls::stream<int64_t> &out_stage) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      int64_t val_a = in_a.read();
      int64_t val_b = in_b.read();
      out_stage << (val_a * 3) - val_b;
   }
}

static void stage2(hls::stream<int64_t> &in_stage, hls::stream<int64_t> &out_stage) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      int64_t prev_result = in_stage.read();
      out_stage << prev_result * (prev_result + 5);
   }
}

static void stage3(hls::stream<int64_t> &in_stage, hls::stream<int32_t> &in_a_passthrough,
                   hls::stream<int64_t> &out_stage) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      int64_t prev_result = in_stage.read();
      int64_t val_a = in_a_passthrough.read();
      int64_t abs_val_a = (val_a < 0) ? -val_a : val_a;
      out_stage << prev_result / (abs_val_a + 1);
   }
}

static void stage4(hls::stream<int64_t> &in_stage, hls::stream<int32_t> &in_b_passthrough,
                   hls::stream<int32_t> &final_result) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      int64_t prev_result = in_stage.read();
      int64_t val_b = in_b_passthrough.read();
      final_result << (int32_t)(prev_result + (val_b * 7));
   }
}

// --- Load/Store Functions ---

static void load_input(int32_t *in, hls::stream<int32_t> &out1, hls::stream<int32_t> &out2) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      int32_t val = in[i];
      out1 << val;
      out2 << val; // Pass-through for later stages
   }
}

static void store_result(int32_t *out, hls::stream<int32_t> &in_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      out[i] = in_stream.read();
   }
}

extern "C" {

/**
 * @brief Top-level kernel that implements a 4-stage deep pipeline.
 */
void krnl_deep_pipeline_calculation(int32_t *in1, int32_t *in2, int32_t *out, int size) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out bundle = gmem0

   // Streams to connect the pipeline stages
   static hls::stream<int32_t> stream_a("stream_a");
   static hls::stream<int32_t> stream_b("stream_b");
   static hls::stream<int32_t> stream_a_passthrough("stream_a_passthrough");
   static hls::stream<int32_t> stream_b_passthrough("stream_b_passthrough");
   static hls::stream<int64_t> stream_stage1_to_2("stream_stage1_to_2");
   static hls::stream<int64_t> stream_stage2_to_3("stream_stage2_to_3");
   static hls::stream<int64_t> stream_stage3_to_4("stream_stage3_to_4");
   static hls::stream<int32_t> stream_to_store("stream_to_store");

#pragma HLS dataflow
   // The dataflow region creates a hardware pipeline from the following function calls
   load_input(in1, stream_a, stream_a_passthrough);
   load_input(in2, stream_b, stream_b_passthrough);
   stage1(stream_a, stream_b, stream_stage1_to_2);
   stage2(stream_stage1_to_2, stream_stage2_to_3);
   stage3(stream_stage2_to_3, stream_a_passthrough, stream_stage3_to_4);
   stage4(stream_stage3_to_4, stream_b_passthrough, stream_to_store);
   store_result(out, stream_to_store);
}
}