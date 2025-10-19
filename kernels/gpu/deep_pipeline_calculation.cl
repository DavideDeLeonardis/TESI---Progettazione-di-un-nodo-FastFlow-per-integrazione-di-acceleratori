/**
 * @brief Esegue un'operazione a pipeline profonda a 4 stadi su due vettori.
 *
 * Per ogni elemento i, questo kernel esegue una sequenza di 4 operazioni
 * matematiche. A differenza della versione FPGA, qui non c'è una pipeline
 * hardware; ogni work-item della GPU esegue l'intera sequenza di calcoli
 * per un singolo elemento.
 *
 * @param a Puntatore al primo vettore di input in memoria globale.
 * @param b Puntatore al secondo vettore di input in memoria globale.
 * @param c Puntatore al vettore di output in memoria globale.
 * @param n Il numero totale di elementi nei vettori.
 */
__kernel void deep_pipeline_calculation(__global const int* a,
                            __global const int* b,
                            __global int* c,
                            const unsigned int n) {

    const int i = get_global_id(0);

    if (i < n) {
        long val_a = a[i];
        long val_b = b[i];

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
        
        c[i] = (int)final_result;
    }
}