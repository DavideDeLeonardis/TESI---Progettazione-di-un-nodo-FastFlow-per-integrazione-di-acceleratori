#include <metal_stdlib>
using namespace metal;

/**
 * @brief Esegue un'operazione a pipeline profonda a 4 stadi su due vettori.
 *
 * Questa è la versione in Metal Shading Language (MSL) del kernel.
 * Ogni thread della GPU esegue l'intera sequenza di 4 operazioni per un
 * singolo elemento, sfruttando il parallelismo dei dati.
 *
 * @param a         Puntatore al primo vettore di input [attributo buffer(0)].
 * @param b         Puntatore al secondo vettore di input [attributo buffer(1)].
 * @param c         Puntatore al vettore di output [attributo buffer(2)].
 * @param n         Il numero totale di elementi nei vettori [attributo buffer(3)].
 * @param gid       L'ID globale del thread, fornito dalla GPU [attributo thread_position_in_grid].
 */
kernel void deep_pipeline_calculation(device const int* a [[buffer(0)]],
                          device const int* b [[buffer(1)]],
                          device int* c       [[buffer(2)]],
                          constant uint& n    [[buffer(3)]],
                          uint gid            [[thread_position_in_grid]])
{
    // Boundary check per evitare accessi a memoria non valida.
    if (gid >= n) {
        return;
    }

    // Usa 'long' (64-bit) per i calcoli intermedi per prevenire l'overflow.
    long val_a = a[gid];
    long val_b = b[gid];

    // --- Inizio della sequenza di operazioni ---

    // Stage 1
    long result_s1 = (val_a * 3) - val_b;

    // Stage 2
    long result_s2 = result_s1 * (result_s1 + 5);
    
    // Stage 3
    long abs_val_a = (val_a < 0) ? -val_a : val_a;
    long result_s3 = result_s2 / (abs_val_a + 1);

    // Stage 4
    long final_result = result_s3 + (val_b * 7);
    
    // Assegna il risultato finale al buffer di output.
    c[gid] = (int)final_result;
}
