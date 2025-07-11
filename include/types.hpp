#pragma once
#include <cstddef>

// Task: puntatori ai tre vettori (host) e dimensione
struct Task {
   int *a;
   int *b;
   int *c;
   size_t n;
};

// Result: qui restituiamo solo il vettore C e n (può essere lo stesso Task)
struct Result {
   int *c;
   size_t n;
};
