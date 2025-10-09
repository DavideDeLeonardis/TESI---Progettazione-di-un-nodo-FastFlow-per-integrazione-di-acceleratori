#pragma once
#include <cstddef>

// Task: puntatori ai tre vettori e dimensione
struct Task {
   int *a, *b, *c;
   size_t n;
};

// Result: restituiamo solo il vettore C e n (può essere lo stesso Task)
struct Result {
   int *c;
   size_t n;
};
