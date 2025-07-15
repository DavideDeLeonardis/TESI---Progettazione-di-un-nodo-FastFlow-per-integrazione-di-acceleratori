__kernel void vecAdd(__global const int* a,
                     __global const int* b,
                     __global int* c,
                     const uint n) {
  uint i = get_global_id(0);
  if (i < n) c[i] = a[i] + b[i];
}
