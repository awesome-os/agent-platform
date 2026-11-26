/*
 * B8G Model Operations & Boundary-Verified MoE C11 ABI (B8G-ABI v3)
 * Zero external headers: uses compiler built-ins / exact-width typedefs only.
 * Verified with: zig cc -std=c11 -Wall -Wextra -Werror -fsyntax-only
 */
#ifndef B8G_MODEL_OPS_ABI_H
#define B8G_MODEL_OPS_ABI_H

typedef unsigned char      b8g_u8;
typedef unsigned short     b8g_u16;
typedef unsigned int       b8g_u32;
typedef unsigned long long b8g_u64;
typedef float              b8g_f32;

/* 64-byte cache-line aligned Model Header inside B8G_ARENA_V1 */
typedef struct {
  b8g_u32 magic;               /* 'B8GM' = 0x4D473842 */
  b8g_u16 abi_version;         /* 1 */
  b8g_u16 topology_kind;       /* 0=standard, 1=looped_core, 2=asymmetric_tapered */
  b8g_u32 hidden_size;
  b8g_u32 vocab_size;
  b8g_u16 physical_layers;
  b8g_u16 logical_layers;
  b8g_u16 loop_passes;         /* 1 for non-looped, e.g. 3 for Inkling 2 Mini */
  b8g_u16 loop_gate_rank;      /* e.g. 64 */
  b8g_u16 active_experts;      /* e.g. 8 */
  b8g_u16 total_experts;       /* e.g. 128 */
  b8g_u16 num_q_heads;
  b8g_u16 num_kv_heads;
  b8g_u32 max_context_tokens;
  b8g_u32 quant_profile_flags; /* bitmask: FP16/MXFP8/NVFP4/Q4KM/BITNET/LITTLEBIT */
  b8g_u64 estimated_bytes;
  b8g_u64 reserved0;
  b8g_u64 reserved1;
} B8gModelHeaderV1;

/* 32-byte Boundary-Verified MoE Route Descriptor */
typedef struct {
  b8g_u32 expert_id;
  b8g_u32 tenant_scope_hash;
  b8g_f32 router_weight;
  b8g_f32 ingress_l2_norm;
  b8g_f32 egress_l2_norm;
  b8g_u32 gate_status;         /* 0=OK, 1=CAP_DENIED, 2=NAN_INF, 3=NORM_EXCEEDED, 4=SYMBOLIC_REJECT */
  b8g_u32 loop_pass_index;
  b8g_u32 flags;
} B8gMoeRouteDescriptorV1;

/* 32-byte Gate Audit Telemetry Record */
typedef struct {
  b8g_u64 total_dispatches;
  b8g_u64 quarantined_experts;
  b8g_u64 capability_denials;
  b8g_u32 active_k_avg_x100;
  b8g_u32 max_pass_count;
} B8gBoundaryGateAuditV1;

_Static_assert(sizeof(B8gModelHeaderV1) == 64, "B8gModelHeaderV1 must be 64 bytes (1 cache line)");
_Static_assert(sizeof(B8gMoeRouteDescriptorV1) == 32, "B8gMoeRouteDescriptorV1 must be 32 bytes");
_Static_assert(sizeof(B8gBoundaryGateAuditV1) == 32, "B8gBoundaryGateAuditV1 must be 32 bytes");

#endif /* B8G_MODEL_OPS_ABI_H */
