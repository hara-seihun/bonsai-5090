#pragma once
// Whole-token tuning knobs. A tuner-generated header (tune/cfg/<id>.hpp) replaces
// these defaults when HALO_TUNE_CONFIG names it. Every knob selects among bodies
// that are bit-identical to the reference engine; numerical variants never live
// here. Owner: halo-autotune (tune/README.md).
#include <cstdint>
// Genes added after a configuration was generated fall back to these values:
// qualified lookup of tune::X only searches tune_defaults when tune itself
// declares no X.
namespace halo6000::tune_defaults {
static constexpr int fuse_norm=0;     // 1: replicate input/post norm+Q8 in every CTA, fused into projection/router
static constexpr int fuse_select=0;   // 1: replicate top-8 selection in every CTA, fused into gate/up
static constexpr int prefetch=0;      // L2 prefetch mask, see schedule.cu
static constexpr int mma_proj=0;      // 1: GDN QKV/Z and attention Q/K/V through the prepared INT8 MMA image
static constexpr int mma_out=0;       // 1: output projection through the prepared MMA image
static constexpr int mma_head=0;      // 1: vocabulary head through the prepared MMA image
static constexpr int proj_rows=1;     // rows per lane group in flight (projections), canonical tree per row
static constexpr int out_rows=1;      // same for the output projection
static constexpr int router_mrows=1;  // F32 router rows per warp in flight
static constexpr int down_rows=1;     // routed+shared down output rows per warp in flight (PANE only)
static constexpr int load_nc=0;       // 1: weight words/scales through ld.global.nc
static constexpr int router_bf16=0;   // 1: router from a load-time RNE BF16 image (lossless iff every coefficient's low 16 bits are 0, true for v1; guard decides)
static constexpr int l2_router=0;     // 1: router kernels carry a persisting L2 access window over their router matrix
static constexpr int l2_state=0;      // 1: recurrence kernels carry a persisting window over their layer's GDN state
static constexpr int l2_ratio=100;    // window hitRatio percent
static constexpr int segment_grids[32]={}; // per class default grid (0: min_blocks x SMs; <0: clamp to residency)
}
namespace halo6000::tune { using namespace halo6000::tune_defaults; }
#ifdef HALO_TUNE_CONFIG
#include HALO_TUNE_CONFIG
#else
namespace halo6000::tune {
static constexpr const char*config_id="default";
// Row ownership (row_exact<L> virtualizes the original 32-lane tree).
static constexpr int qkv_lanes=16;     // GDN QKV rows, attention Q+gate rows
static constexpr int z_lanes=16;       // GDN Z rows, attention K/V rows
static constexpr int ab_lanes=16;      // GDN alpha/beta rows
static constexpr int out_lanes=16;     // GDN/attention output projection
static constexpr int router_rows=2;    // F32 router rows per CTA (0 = generic one-warp router)
static constexpr int select_body=1;    // 0 serial top8, 1 warp-bitonic candidate
static constexpr int head_lanes=16;    // head_q8_phase<L>: 4/8/16/32 keep the sixteen-group tree
static constexpr int head_contiguous=0;// striped (0) or contiguous (1) tile ownership
// Segments: one entry per compiled kernel class. kinds = bit mask of EpochKind,
// min_blocks = __launch_bounds__ second argument.
static constexpr int segment_count=1;
static constexpr uint32_t segment_kinds[]={0x7fefbfu}; // all kinds except rotation
static constexpr int segment_min_blocks[]={1};
// kind bit set => a kernel boundary precedes every epoch of that kind.
static constexpr uint32_t boundary_kinds=0;
}
#endif
