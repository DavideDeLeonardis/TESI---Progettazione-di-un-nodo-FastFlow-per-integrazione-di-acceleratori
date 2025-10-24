#include <metal_stdlib>
using namespace metal;

/**
 * @brief Esegue un calcolo computazionalmente intensivo (versione MSL).
 *
 * Per ogni elemento, esegue un ciclo di 100 iterazioni di calcoli
 * trigonometrici (sin, cos) per stressare le unità di calcolo.
 *
 * @param a         Puntatore al primo vettore di input [buffer(0)].
 * @param b         Puntatore al secondo vettore di input [buffer(1)].
 * @param c         Puntatore al vettore di output [buffer(2)].
 * @param n         Il numero totale di elementi [buffer(3)].
 * @param gid       L'ID globale del thread.
 */
kernel void heavy_compute_kernel(device const int* a [[buffer(0)]],
                                 device const int* b [[buffer(1)]],
                                 device int* c       [[buffer(2)]],
                                 constant uint& n    [[buffer(3)]],
                                 uint gid            [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }

    // Converte gli input in float per le funzioni trigonometriche
    float val_a = (float)a[gid];
    float val_b = (float)b[gid];
    float result = 0.0f;

    // Ciclo computazionalmente pesante
    for (int j = 0; j < 100; ++j) {
        result += sin(val_a + j) * cos(val_b - j);
    }

    // Riconverte il risultato finale in int
    c[gid] = (int)result;
}