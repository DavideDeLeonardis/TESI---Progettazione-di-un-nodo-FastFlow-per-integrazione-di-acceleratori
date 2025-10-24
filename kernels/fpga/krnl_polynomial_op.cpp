//
// Kernel
//

/*******************************************************************************
Description:

    This kernel implements a complex polynomial operation using the
    load/compute/store coding style for Vitis HLS. It follows a dataflow
    architecture where data is streamed between three distinct stages:
    1. load_input: Reads data from global memory into streams.
    2. compute_poly: Performs the core polynomial calculation.
    3. store_result: Writes the results from a stream back to global memory.

    The operation performed is:
    c[i] = (2 * a[i]^2) + (3 * a[i]^3) - (4 * b[i]^2) + (5 * b[i]^5)

    This structure allows for task-level pipelining, enabling the stages
    to operate in parallel for maximum throughput.

*******************************************************************************/

// Includes
#include <hls_stream.h>
#include <stdint.h>

#define DATA_SIZE 4096

// TRIPCOUNT identifier
const int c_size = DATA_SIZE;

/**
 * @brief Reads data from global memory and writes it into an HLS stream.
 * * @param in Pointer to the input vector in global memory.
 * @param inStream The output HLS stream.
 * @param size The number of elements to process.
 */
static void load_input(int32_t *in, hls::stream<int32_t> &inStream, int size) {
mem_rd:
   for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
      inStream << in[i];
   }
}

/**
 * @brief Reads from input streams, performs the polynomial calculation, and
 * writes to an output stream.
 * * @param in1_stream Stream for the first input vector 'a'.
 * @param in2_stream Stream for the second input vector 'b'.
 * @param out_stream The output stream for the results 'c'.
 * @param size The number of elements to process.
 */
static void compute_poly(hls::stream<int32_t> &in1_stream,
                         hls::stream<int32_t> &in2_stream,
                         hls::stream<int32_t> &out_stream, int size) {
execute:
   for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
      int32_t val_a = in1_stream.read();
      int32_t val_b = in2_stream.read();

      // Calculate powers using explicit multiplications for hardware
      // efficiency. Use 64-bit integers for intermediate products to prevent
      // overflow, as powers (especially b^5) can exceed the range of a 32-bit
      // integer.
      int64_t a2 = (int64_t)val_a * val_a;
      int64_t a3 = a2 * val_a;
      int64_t b2 = (int64_t)val_b * val_b;
      int64_t b4 = b2 * b2;
      int64_t b5 = b4 * val_b;

      // Perform the final polynomial calculation using 64-bit integers
      // to ensure correctness during intermediate additions/subtractions.
      int64_t result = (2 * a2) + (3 * a3) - (4 * b2) + (5 * b5);

      out_stream << (int32_t)result;
   }
}

/**
 * @brief Reads data from an HLS stream and writes it to global memory.
 * * @param out Pointer to the output vector in global memory.
 * @param out_stream The input HLS stream.
 * @param size The number of elements to process.
 */
static void store_result(int32_t *out, hls::stream<int32_t> &out_stream,
                         int size) {
mem_wr:
   for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
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
void krnl_polynomial_op(int32_t *in1, int32_t *in2, int32_t *out, int size) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = out bundle = gmem0

   static hls::stream<int32_t> in1_stream("input_stream_1");
   static hls::stream<int32_t> in2_stream("input_stream_2");
   static hls::stream<int32_t> out_stream("output_stream");

#pragma HLS dataflow
   // dataflow pragma instructs compiler to run following three APIs in parallel
   load_input(in1, in1_stream, size);
   load_input(in2, in2_stream, size);
   compute_poly(in1_stream, in2_stream, out_stream, size);
   store_result(out, out_stream, size);
}
}
