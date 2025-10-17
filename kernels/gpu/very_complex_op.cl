/**
 * @brief Esegue un'operazione computazionalmente intensiva e dipendente dai dati.
 *
 * Per ogni elemento i, questo kernel esegue un calcolo iterativo per stressare
 * le unità di calcolo dell'acceleratore. La logica include potenze, un ciclo
 * interno e operazioni condizionali implicite (protezione dalla divisione per zero).
 *
 * @param a Puntatore al primo vettore di input in memoria globale.
 * @param b Puntatore al secondo vettore di input in memoria globale.
 * @param c Puntatore al vettore di output in memoria globale.
 * @param n Il numero totale di elementi nei vettori.
 */
__kernel void very_complex_op(__global const int* a,
                              __global const int* b,
                              __global int* c,
                              const unsigned int n) {

    const int i = get_global_id(0);

    if (i < n) {
        long val_a = a[i];
        long val_b = b[i];

        long result = (3 * val_a * val_a) - (2 * val_b * val_b);

        for (int j = 0; j < 5; ++j) {
            long divisor = abs(val_b) + 1;
            result = (result * (val_a + j)) / divisor + (val_b * (j + 1));
        }

        result += (val_a - val_b) * 7;
        
        c[i] = (int)result;
    }
}