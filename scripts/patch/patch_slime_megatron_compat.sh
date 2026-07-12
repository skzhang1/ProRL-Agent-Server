#!/usr/bin/env bash
# Keep Slime v0.3.0 compatible with the pinned Megatron-LM 26.04 checkout.
set -euo pipefail

slime_dir="${SLIME_DIR:?SLIME_DIR is required}"
target="${slime_dir}/slime/backends/megatron_utils/arguments.py"
[ -f "${target}" ] || { echo "missing Slime arguments module: ${target}" >&2; exit 1; }

python3 - "${target}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
marker = '    # Compatibility for Megatron-LM versions whose parser uses\n'
block = '''    # Compatibility for Megatron-LM versions whose parser uses
    # use_gloo_process_groups instead of Slime's expected
    # enable_gloo_process_groups attribute.
    if not hasattr(args, "enable_gloo_process_groups"):
        args.enable_gloo_process_groups = getattr(args, "use_gloo_process_groups", False)

    # --norm-epsilon maps to layernorm_epsilon in newer Megatron-LM.
    if not hasattr(args, "norm_epsilon"):
        args.norm_epsilon = getattr(args, "layernorm_epsilon", 1e-5)

    # Disable TE-dependent features when Transformer Engine is not installed.
    try:
        import transformer_engine  # noqa: F401
    except ImportError:
        if getattr(args, "apply_rope_fusion", None) is not False:
            args.apply_rope_fusion = False

'''
anchor = '''    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

'''
if marker in text:
    print(f"Slime/Megatron compatibility already applied: {path}")
elif anchor in text:
    path.write_text(text.replace(anchor, anchor + block, 1), encoding="utf-8")
    print(f"Patched Slime/Megatron compatibility: {path}")
else:
    raise SystemExit(f"unexpected Slime arguments layout: {path}")
PY
