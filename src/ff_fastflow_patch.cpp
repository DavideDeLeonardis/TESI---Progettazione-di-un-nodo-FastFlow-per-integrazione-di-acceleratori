// // Patch minima: fornisce le definizioni mancanti in FastFlow header-only
// #include <ff/pipeline.hpp> // include una sola volta è sufficiente

// namespace ff {

// // --- definizioni neutre --------------------------------------------
// static bool isfarm_withcollector(ff_node *) { return false; }
// static bool isfarm_multimultioutput(ff_node *) { return false; }

// static const svector<ff_node *> &dummy() {
//    static svector<ff_node *> d;
//    return d;
// }
// static const svector<ff_node *> &isa2a_getfirstset(ff_node *) {
//    return dummy();
// }
// static const svector<ff_node *> &isa2a_getsecondset(ff_node *) {
//    return dummy();
// }
// static const svector<ff_node *> &isfarm_getworkers(ff_node *) {
//    return dummy();
// }

// static ff_node *ispipe_getlast(ff_node *) { return nullptr; }

// } 
