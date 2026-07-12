#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/opt/polr_venv/bin/python}"
"${python_bin}" - <<'PY'
from pathlib import Path

import sglang

if sglang.__version__ != "0.5.13":
    raise SystemExit(f"expected SGLang 0.5.13, got {sglang.__version__}")

path = (
    Path(sglang.__file__).resolve().parent
    / "srt"
    / "entrypoints"
    / "openai"
    / "serving_chat.py"
)
tokenizer_manager_path = (
    Path(sglang.__file__).resolve().parent
    / "srt"
    / "managers"
    / "tokenizer_manager.py"
)
text = path.read_text(encoding="utf-8")
tokenizer_text = tokenizer_manager_path.read_text(encoding="utf-8")

# SGLang 0.5.13 already has ReqState.prompt_token_ids, but fills it only
# when the router preserves return_prompt_token_ids. Capture canonical IDs
# unconditionally and place them in meta_info, which the router preserves.
capture_blocks = (
    (
        "                if obj.return_prompt_token_ids:\n"
        "                    state.prompt_token_ids = list(tokenized_obj.input_ids)\n",
        "                state.prompt_token_ids = list(tokenized_obj.input_ids)\n",
    ),
    (
        "                    if tmp_obj.return_prompt_token_ids:\n"
        "                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)\n",
        "                    state.prompt_token_ids = list(tokenized_objs[i].input_ids)\n",
    ),
    (
        "                        if tmp_obj.return_prompt_token_ids:\n"
        "                            state.prompt_token_ids = list(tokenized_obj.input_ids)\n",
        "                        state.prompt_token_ids = list(tokenized_obj.input_ids)\n",
    ),
)
for old, new in capture_blocks:
    if old in tokenizer_text:
        tokenizer_text = tokenizer_text.replace(old, new)
    elif new not in tokenizer_text:
        raise SystemExit(f"unexpected prompt capture block in {tokenizer_manager_path}")

meta_anchor = '''            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.server_args.weight_version,
                "num_retractions": recv_obj.retraction_counts[i],
            }
'''
meta_patched = meta_anchor + '''            if state.prompt_token_ids is not None:
                meta_info["input_token_ids"] = list(state.prompt_token_ids)
'''
if meta_patched not in tokenizer_text:
    if meta_anchor not in tokenizer_text:
        raise SystemExit(f"unexpected meta_info block in {tokenizer_manager_path}")
    tokenizer_text = tokenizer_text.replace(meta_anchor, meta_patched, 1)

tokenizer_manager_path.write_text(tokenizer_text, encoding="utf-8")

old_generate = "            return_prompt_token_ids=request.return_prompt_token_ids,\n"
new_generate = "            return_prompt_token_ids=True,  # Required by Polar training.\n"

old_prompt = '''            choice_prompt_token_ids = (
                ret_item.get("prompt_token_ids")
                if request.return_prompt_token_ids
                else None
            )
'''
new_prompt = '''            # Polar training requires exact prompt IDs even when an
            # intermediate OpenAI router drops SGLang extension flags.
            choice_prompt_token_ids = ret_item.get("prompt_token_ids")
'''
old_meta = '''            choice_meta_info = (
                ret_item["meta_info"] if request.return_meta_info else None
            )
'''
new_meta = '''            # Preserve generated token IDs and rollout logprobs for
            # agentic trajectory construction. The router keeps meta_info but
            # drops the top-level prompt_token_ids extension.
            choice_meta_info = dict(ret_item["meta_info"])
            choice_meta_info["prompt_token_ids"] = ret_item.get("prompt_token_ids")
'''

if new_generate not in text:
    if old_generate not in text:
        raise SystemExit(f"unexpected GenerateReq prompt-token field in {path}")
    text = text.replace(old_generate, new_generate, 1)
if new_prompt not in text:
    if old_prompt not in text:
        raise SystemExit(f"unexpected prompt-token block in {path}")
    text = text.replace(old_prompt, new_prompt, 1)
if new_meta not in text:
    if old_meta not in text:
        raise SystemExit(f"unexpected meta-info block in {path}")
    text = text.replace(old_meta, new_meta, 1)

path.write_text(text, encoding="utf-8")
print(f"patched SGLang token metadata response: {path}")
PY
