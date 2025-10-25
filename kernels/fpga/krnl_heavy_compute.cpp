/*******************************************************************************
Description:

    This kernel implements a computationally-intensive (compute-bound)
    operation using the load/compute/store coding style for Vitis HLS.
    It follows a dataflow architecture where data is streamed between
    three distinct stages:
    1. load_input: Reads data from global memory into streams.
    2. compute_heavy: Performs the core iterative trigonometric calculation.
    3. store_result: Writes the results from a stream back to global memory.

    The operation performed is a 100-iteration loop of sin/cos calculations
    for each element, designed to make the kernel compute-bound.

*******************************************************************************/

// Includes
#include <hls_math.h>
#include <hls_stream.h>
#include <stdint.h>

#define DATA_SIZE 4096

// TRIPCOUNT identifier
const int c_size = DATA_SIZE;

/**
 * @brief Reads data from global memory and writes it into an HLS stream.
 */
static void load_input(int32_t *in, hls::stream<int32_t> &inStream, int size) {
mem_rd:
   for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      inStream << in[i];
   }
}

/**
 * @brief Reads from input streams, performs the heavy trigonometric calculation,
 * and writes to an output stream.
 */
static void compute_heavy(hls::stream<int32_t> &in1_stream, hls::stream<int32_t> &in2_stream,
                          hls::stream<int32_t> &out_stream, int size) {
execute:
   for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size

      double val_a = (double)in1_stream.read();
      double val_b = (double)in2_stream.read();
      double result = 0.0;

   // Ciclo computazionalmente pesante (200 iterazioni)
   // Vitis HLS ottimizzerà questo ciclo interno.
   compute_loop:
      for (int j = 0; j < 200; ++j) {
#pragma HLS PIPELINE // Applichiamo la pipeline al LOOP INTERNO
         result += hls::sin(val_a + j) * hls::cos(val_b - j);
      }

      out_stream << (int32_t)result;
   }
}

/**
 * @brief Reads data from an HLS stream and writes it to global memory.
 */
static void store_result(int32_t *out, hls::stream<int32_t> &out_stream, int size) {
mem_wr:
   for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II = 1
      out[i] = out_stream.read();
   }
}

extern "C" {

/**
 * @brief Top-level kernel function that orchestrates the dataflow pipeline.
 *
 * @param in1  (input)  --> Input vector 'a'
 * @param in2  (input)  --> Input vector 'b'
 * @param out  (output) --> Output vector 'c'
 * @param size (input)  --> Number of elements in vectors
 */
void krnl_heavy_compute(int32_t *in1, int32_t *in2, int32_t *out, int size) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out bundle = gmem0

   static hls::stream<int32_t> in1_stream("input_stream_1");
   static hls::stream<int32_t> in2_stream("input_stream_2");
   static hls::stream<int32_t> out_stream("output_stream");

#pragma HLS dataflow
   // dataflow pragma instruct compiler to run following three APIs in parallel
   load_input(in1, in1_stream, size);
   load_input(in2, in2_stream, size);
   compute_heavy(in1_stream, in2_stream, out_stream, size);
   store_result(out, out_stream, size);
}
}