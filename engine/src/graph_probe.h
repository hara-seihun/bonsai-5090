#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { H5_GRAPH_UPLOAD_AT_LAUNCH = 0, H5_GRAPH_UPLOAD_AT_CAPTURE = 1 } H5GraphUpload;
typedef enum { H5_GRAPH_TIMING_OFF = 0, H5_GRAPH_TIMING_WALL_THREAD = 1 } H5GraphTiming;
typedef enum { H5_GRAPH_NODES_NATIVE = 0, H5_GRAPH_NODES_KERNEL_MEMORY = 1 } H5GraphNodes;
typedef enum {
  H5_GRAPH_PROBE_OK = 0, H5_GRAPH_PROBE_BAD_ARGUMENT = -1,
  H5_GRAPH_PROBE_ALREADY_USED = -2, H5_GRAPH_PROBE_DISABLED = -3,
  H5_GRAPH_PROBE_NO_SAMPLE = -4, H5_GRAPH_PROBE_GRAPHS_DISABLED = -5
} H5GraphProbeError;
typedef enum {
  H5_GRAPH_ROUTE_EAGER = 0, H5_GRAPH_ROUTE_CAPTURED = 1,
  H5_GRAPH_ROUTE_CACHED = 2, H5_GRAPH_ROUTE_MEGA = 3
} H5GraphRoute;

typedef struct {
  H5GraphRoute route;
  int shape[8];
  uint64_t nodes, roots;
  uint64_t prepare_ns, submit_ns, launch_wall_ns, launch_thread_ns;
  uint64_t sync_ns, finish_ns, total_ns, upload_wall_ns;
  uint64_t rewrite_ns, memory_copies, memory_clears, copy_bytes, clear_bytes;
  uint64_t host_read_bytes, host_write_bytes;
} H5GraphSample;

H5GraphProbeError h5_graph_probe_configure(void*, H5GraphUpload, H5GraphTiming);
H5GraphProbeError h5_graph_probe_read(void*, H5GraphSample*);
H5GraphProbeError h5_graph_nodes_configure(void*, H5GraphNodes);

#ifdef __cplusplus
}
#endif
