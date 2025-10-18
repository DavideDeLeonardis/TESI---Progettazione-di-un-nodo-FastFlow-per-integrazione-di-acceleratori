#include <metal_stdlib>
using namespace metal;

/**
 * @brief Esegue la somma elemento per elemento di due vettori (a + b = c).
 *
 * Questa è la versione in Metal Shading Language (MSL) del kernel.
 *
 * @param a         Puntatore al primo vettore di input [attributo buffer(0)].
 * @param b         Puntatore al secondo vettore di input [attributo buffer(1)].
 * @param c         Puntatore al vettore di output [attributo buffer(2)].
 * @param n         Il numero totale di elementi nei vettori [attributo buffer(3)].
 * @param gid       L'ID globale del thread, fornito dalla GPU [attributo thread_position_in_grid].
 */
kernel void vecAdd(device const int* a [[buffer(0)]],
                   device const int* b [[buffer(1)]],
                   device int* c       [[buffer(2)]],
                   constant uint& n    [[buffer(3)]],
                   uint gid            [[thread_position_in_grid]])
{
    // Esegue un controllo per assicurarsi che il thread non acceda a memoria
    // fuori dai limiti del vettore.
    if (gid >= n) {
        return;
    }

    // Esegue la somma elemento per elemento.
    c[gid] = a[gid] + b[gid];
}
