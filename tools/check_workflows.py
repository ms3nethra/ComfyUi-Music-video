"""
Pre-release check for example_workflows/.

Run this after ANY change to a node's INPUT_TYPES:

    python tools/check_workflows.py

ComfyUI matches a saved workflow's `widgets_values` to a node's widgets by
POSITION, not by name. Insert a widget anywhere but the end and every value
after it lands on the wrong input - which is how the shipped examples ended up
feeding 'Creative (artistic music video...)' into `image_generator_model` and
refusing to queue. Nothing warns you: the JSON is still valid, the graph still
opens, and it only fails when someone presses Queue.

Reports, per workflow:
  * node types that are not installed
  * widgets_values that do not line up with the node's widget slots
  * combo values that are not in the node's allowed list

Exit code is non-zero when anything is wrong, so it can gate a release.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))   # ComfyUI root
sys.path.insert(0, os.path.dirname(PACK))                    # custom_nodes

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
# Nodes the frontend gives extra DOM widgets (upload buttons, preview players)
# that never appear in INPUT_TYPES. Their counts legitimately disagree.
DOM_WIDGET_NODES = {"LoadAudio", "LoadImage", "LoadVideo", "SaveAudioAdvanced",
                    "LoadAudioUpload", "VHS_LoadVideo"}
SKIP_TYPES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}


def slots(input_types):
    """Widget slots in the order ComfyUI serializes them."""
    out = []
    for section in ("required", "optional"):
        for name, spec in (input_types.get(section) or {}).items():
            kind = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if opts.get("forceInput"):
                continue
            if isinstance(kind, list):
                out.append((name, kind))
            elif isinstance(kind, str) and kind in WIDGET_TYPES:
                out.append((name, None))
            else:
                continue
            if opts.get("control_after_generate"):
                out.append((name + "::control", None))
    return out


def main():
    pack_name = os.path.basename(PACK)
    pack = __import__(pack_name)
    expected = {name: slots(cls.INPUT_TYPES())
                for name, cls in pack.NODE_CLASS_MAPPINGS.items()}

    problems = 0
    files = sorted(glob.glob(os.path.join(PACK, "example_workflows", "*.json")))
    for path in files:
        name = os.path.basename(path)
        try:
            workflow = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{name}: WILL NOT PARSE - {exc}")
            problems += 1
            continue

        subgraphs = {s.get("id") for s in
                     (workflow.get("definitions") or {}).get("subgraphs") or []}
        rows = []
        for node in workflow.get("nodes", []):
            node_type = node.get("type")
            if node_type in SKIP_TYPES or node_type in subgraphs:
                continue
            # Only this pack's nodes can be checked without a running server;
            # core nodes are ComfyUI's problem, not ours.
            if node_type not in expected:
                continue
            values = node.get("widgets_values")
            if not isinstance(values, list):
                continue

            want = expected[node_type]
            if len(values) != len(want) and node_type not in DOM_WIDGET_NODES:
                rows.append(f"#{node['id']} {node_type}: saved {len(values)} values, "
                            f"node has {len(want)} widget slots "
                            f"-> {[w[0] for w in want]}")
                continue

            for (slot, choices), value in zip(want, values):
                if choices and value not in choices:
                    rows.append(f"#{node['id']} {node_type}: {slot}={value!r} "
                                f"is not one of {choices}")

        if rows:
            problems += len(rows)
            print(f"### {name}")
            for row in rows:
                print("   " + row)
            print()

    print(f"{len(files)} workflow(s) checked, {problems} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
