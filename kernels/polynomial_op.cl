/**
 * @brief Esegue un'operazione polinomiale complessa su due vettori di input.
 *
 * Per ogni elemento i, calcola: 
 * c[i] = (2 * a[i]^2) + (3 * a[i]^3) - (4 * b[i]^2) + (5 * b[i]^5)
 *
 * @param a Puntatore al primo vettore di input in memoria globale.
 * @param b Puntatore al secondo vettore di input in memoria globale.
 * @param c Puntatore al vettore di output in memoria globale.
 * @param n Il numero totale di elementi nei vettori.
 */
__kernel void polynomial_op(__global const int* a,
                         __global const int* b,
                         __global int* c,
                         const unsigned int n) {
    const int i = get_global_id(0);

    if (i < n) {
        int val_a = a[i];
        int val_b = b[i];

        int a2 = val_a * val_a;
        int a3 = a2 * val_a;
        int b2 = val_b * val_b;
        int b4 = b2 * b2;
        int b5 = b4 * val_b;

        c[i] = (2 * a2) + (3 * a3) - (4 * b2) + (5 * b5);
    }
}
