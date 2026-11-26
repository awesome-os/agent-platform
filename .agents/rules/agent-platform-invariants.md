# AwesomeOS Agent Platform — Core Invariants

1. **Zero External Runtime Bloat:** Strictly `erasableSyntaxOnly: true` native Node 22+/26 TypeScript (`.mts`) using built-in `node:*` modules only.
2. **Holo v11 & ABI v3:** Content-addressable assets use SHA-512 `v11/files/<aa>/<sha512>` and `B8G_ABI_MAGIC_V3` (`0x33564742`). Zero legacy `b8g-ghost` or `cas-v1` references.
3. **Boundary-Verified Safety:** All MoE experts, ZER0Q dispatches, and Model Armor checks fail closed under non-finite values, norm bounds, prompt injection, or capability mismatch.
