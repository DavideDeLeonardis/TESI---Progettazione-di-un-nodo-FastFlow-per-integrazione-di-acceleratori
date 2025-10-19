/*******************************************************************************
Description:

    This kernel demonstrates a deep pipeline with 4 stages, implemented
    using the load/compute/store style for Vitis HLS. This version has been
    refactored to prevent dataflow deadlocks by passing data linearly
    between stages using a struct (relay pattern), instead of long-distance
    passthrough streams.

*******************************************************************************/

// Includes
#include <hls_stream.h>
#include <stdint.h>

#define DATA_SIZE 4096
const int c_size = DATA_SIZE;

/**
 * @brief Struct per trasportare i dati attraverso la pipeline.
 *
 * Contiene il risultato intermedio del calcolo e i valori originali
 * di 'a' e 'b' che sono necessari negli stadi successivi.
 */
struct PipelineData {
   int64_t result;
   int32_t val_a;
   int32_t val_b;
};

// --- Load/Store Functions ---

static void load_inputs(int32_t *in1, int32_t *in2, hls::stream<PipelineData> &out_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      PipelineData data;
      data.val_a = in1[i];
      data.val_b = in2[i];
      data.result = 0;
      out_stream << data;
   }
}

static void store_result(int32_t *out, hls::stream<int32_t> &in_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      out[i] = in_stream.read();
   }
}

// --- Pipeline Stages ---

static void stage1(hls::stream<PipelineData> &in_stream, hls::stream<PipelineData> &out_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      PipelineData data = in_stream.read();
      data.result = ((int64_t)data.val_a * 3) - data.val_b;
      out_stream << data;
   }
}

static void stage2(hls::stream<PipelineData> &in_stream, hls::stream<PipelineData> &out_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      PipelineData data = in_stream.read();
      data.result = data.result * (data.result + 5);
      out_stream << data;
   }
}

static void stage3(hls::stream<PipelineData> &in_stream, hls::stream<PipelineData> &out_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      PipelineData data = in_stream.read();
      int64_t abs_val_a = (data.val_a < 0) ? -(int64_t)data.val_a : (int64_t)data.val_a;
      data.result = data.result / (abs_val_a + 1);
      out_stream << data;
   }
}

static void stage4(hls::stream<PipelineData> &in_stream, hls::stream<int32_t> &out_stream) {
   for (int i = 0; i < c_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      PipelineData data = in_stream.read();
      int64_t final_result = data.result + ((int64_t)data.val_b * 7);
      out_stream << (int32_t)final_result;
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
   static hls::stream<PipelineData> stream_load_to_s1("stream_load_to_s1");
   static hls::stream<PipelineData> stream_s1_to_s2("stream_s1_to_s2");
   static hls::stream<PipelineData> stream_s2_to_s3("stream_s2_to_s3");
   static hls::stream<PipelineData> stream_s3_to_s4("stream_s3_to_s4");
   static hls::stream<int32_t> stream_s4_to_store("stream_s4_to_store");

#pragma HLS dataflow
   // La regione dataflow crea una pipeline hardware lineare dalle seguenti chiamate
   load_inputs(in1, in2, stream_load_to_s1);
   stage1(stream_load_to_s1, stream_s1_to_s2);
   stage2(stream_s1_to_s2, stream_s2_to_s3);
   stage3(stream_s2_to_s3, stream_s3_to_s4);
   stage4(stream_s3_to_s4, stream_s4_to_store);
   store_result(out, stream_s4_to_store);
}
}