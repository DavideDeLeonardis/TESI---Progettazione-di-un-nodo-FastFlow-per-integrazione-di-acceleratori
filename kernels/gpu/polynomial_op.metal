#include <metal_stdlib>
using namespace metal;

/**
 * @brief Esegue un'operazione polinomiale complessa su due vettori di input.
 *
 * Versione in Metal Shading Language (MSL) del kernel.
 * Per ogni elemento i, calcola: 
 * c[i] = (2 * a[i]^2) + (3 * a[i]^3) - (4 * b[i]^2) + (5 * b[i]^5)
 *
 * @param a         Puntatore al primo vettore di input [attributo buffer(0)].
 * @param b         Puntatore al secondo vettore di input [attributo buffer(1)].
 * @param c         Puntatore al vettore di output [attributo buffer(2)].
 * @param n         Il numero totale di elementi nei vettori [attributo buffer(3)].
 * @param gid       L'ID globale del thread, fornito dalla GPU [attributo thread_position_in_grid].
 */
kernel void polynomial_op(device const int* a [[buffer(0)]],
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

    // Calcolo delle potenze tramite moltiplicazioni esplicite.
    long a2 = val_a * val_a;
    long a3 = a2 * val_a;
    long b2 = val_b * val_b;
    long b4 = b2 * b2;
    long b5 = b4 * val_b;

    // Esegue il calcolo polinomiale finale.
    long result = (2 * a2) + (3 * a3) - (4 * b2) + (5 * b5);
    
    // Assegna il risultato finale, riconvertendolo a int.
    c[gid] = (int)result;
}