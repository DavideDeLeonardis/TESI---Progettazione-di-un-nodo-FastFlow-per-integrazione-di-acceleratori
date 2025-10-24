/**
 * @brief Esegue un calcolo computazionalmente intensivo (compute-bound).
 *
 * Per ogni elemento, esegue un ciclo di 200 iterazioni di calcoli
 * trigonometrici (sin, cos) per stressare le unità di calcolo.
 *
 * @param a Puntatore al primo vettore di input in memoria globale.
 * @param b Puntatore al secondo vettore di input in memoria globale.
 * @param c Puntatore al vettore di output in memoria globale.
 * @param n Il numero totale di elementi nei vettori.
 */
__kernel void heavy_compute_kernel(__global const int* a,
                                   __global const int* b,
                                   __global int* c,
                                   const unsigned int n) {

    const int i = get_global_id(0);

    if (i < n) {
        // Converte gli input in float per le funzioni trigonometriche
        float val_a = (float)a[i];
        float val_b = (float)b[i];
        float result = 0.0f;

        // Ciclo computazionalmente pesante
        for (int j = 0; j < 200; ++j) {
            result += sin(val_a + j) * cos(val_b - j);
        }

        // Riconverte il risultato finale in int
        c[i] = (int)result;
    }
}