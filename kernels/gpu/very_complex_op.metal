#include <metal_stdlib>
using namespace metal;

/**
 * @brief Esegue un'operazione computazionalmente intensiva e dipendente dai dati.
 *
 * Questa è la versione in Metal Shading Language (MSL) del kernel.
 * Per ogni elemento, esegue un calcolo iterativo per stressare le unità di calcolo
 * dell'acceleratore. La logica include potenze, un ciclo interno e operazioni
 * condizionali.
 *
 * @param a         Puntatore al primo vettore di input [attributo buffer(0)].
 * @param b         Puntatore al secondo vettore di input [attributo buffer(1)].
 * @param c         Puntatore al vettore di output [attributo buffer(2)].
 * @param n         Il numero totale di elementi nei vettori [attributo buffer(3)].
 * @param gid       L'ID globale del thread, fornito dalla GPU [attributo thread_position_in_grid].
 */
kernel void very_complex_op(device const int* a [[buffer(0)]],
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

    // Usa 'long' (64-bit) per i calcoli intermedi per prevenire l'overflow.
    long val_a = a[gid];
    long val_b = b[gid];

    // Calcolo iniziale
    long result = (3 * val_a * val_a) - (2 * val_b * val_b);

    // Ciclo iterativo per aumentare il carico computazionale.
    // La dipendenza dal risultato precedente e dai dati di input
    // rende l'operazione più complessa per il compilatore e l'hardware.
    for (int j = 0; j < 5; ++j) {
        // Aggiungiamo 1 al divisore per evitare la divisione per zero in modo sicuro.
        long divisor = abs(val_b) + 1;
        result = (result * (val_a + j)) / divisor + (val_b * (j + 1));
    }

    // Operazione finale di combinazione
    result += (val_a - val_b) * 7;
    
    // Assegna il risultato finale, riconvertendolo a int.
    c[gid] = (int)result;
}
