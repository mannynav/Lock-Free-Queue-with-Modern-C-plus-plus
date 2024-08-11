# Lock-Free Queue

This experiment is a work in progress and will be continuously updated. The goal was to first make the computation of the data lock free in order to avoid the slowdowns associated with locking and unlocking, which was achieved and the second was to reduce the idle time of the threads that were working on both heavier workloads and lighter workloads. The results are currently being sent to a csv, where a few metrics can be compared. The test settings can be adjusted and the data is generated artificially by some time consuming, cpu intensive computations.
