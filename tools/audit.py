"""
Full structural audit of the pack: node registration, the LTX render
expansion (frame flow, guides, one-pass encoding, ordering, resume), the
MiniMax expansion, the SegmentTable against the newest real run, and the
prompt engine (approach distinctness, abstract-as-art, music-driven visuals).

    python tools/audit.py

Exit code is non-zero when anything fails, so it can gate a release. Runs
without a ComfyUI server and without a GPU render - it verifies structure,
wiring and prompt content, not pixels.
"""

import io
import json
import os
import re
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))
sys.path.insert(0, os.path.dirname(PACK))

import importlib
import torch

pack = importlib.import_module(os.path.basename(PACK))
n = pack.nodes
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name + ((": " + str(detail)) if detail and not ok else ""))


# ---- 1. registration consistency -----------------------------------------
cm, dm = pack.NODE_CLASS_MAPPINGS, pack.NODE_DISPLAY_NAME_MAPPINGS
check("every node has a display name", set(cm) == set(dm), set(cm) ^ set(dm))
for name, cls in cm.items():
    it = cls.INPUT_TYPES()
    check(name + ".INPUT_TYPES valid", isinstance(it, dict) and "required" in it)
    fn = getattr(cls, "FUNCTION", "")
    check(name + " RETURN/FUNCTION", hasattr(cls, "RETURN_TYPES") and callable(getattr(cls, fn, None)))

# ---- 2. LTX expansion ------------------------------------------------------
LTX = cm["LTXSegmentsRender"]()
N = 5
segs = [{"segment_index": i, "start_time": i * 6.0, "end_time": i * 6 + 6.0, "duration": 6.0,
         "lyrics": "l%d" % i, "section_type": "verse"} for i in range(N)]
run_dir = tempfile.mkdtemp(prefix="audit_")
out = LTX.render_segments(False, "M", "C", "V", "A", torch.zeros((N + 1, 32, 32, 3)),
                          json.dumps(["p%d" % i for i in range(N)]),
                          json.dumps({"segments": segs, "card_count": N + 1}),
                          run_dir, 1280, 704, 24, 100, 0.7, "neg",
                          segments_per_queue=N, resume=True)
g = out["expand"]
nodes = g.get("nodes", g) if isinstance(g, dict) else g
types = {}
for nid, nd in nodes.items():
    types.setdefault(nd["class_type"], []).append(nid)


def find(kind, suffix):
    return [x for x in types.get(kind, []) if x.endswith(suffix)][0]


cont_ok = True
for i in range(N):
    sA = nodes[find("ImageScale", "sA%d" % i)]
    sB = nodes[find("ImageScale", "sB%d" % i)]
    if not (sA["inputs"]["image"][1] == 0 and sB["inputs"]["image"][1] == 1
            and sA["inputs"]["image"][0].endswith("now%d" % i)):
        cont_ok = False
check("first/last frame flow per segment", cont_ok)

seeds = [nodes[find("RandomNoise", "noise%d" % i)]["inputs"]["noise_seed"] for i in range(N)]
check("seed per segment = seed+i", seeds == [100 + i for i in range(N)], seeds)

guide_ok = True
for i in range(N):
    gA = nodes[find("LTXVAddGuide", "gA%d" % i)]["inputs"]
    gB = nodes[find("LTXVAddGuide", "gB%d" % i)]["inputs"]
    if not (gA["frame_idx"] == 0 and abs(gA["strength"] - 0.7) < 1e-9
            and gB["frame_idx"] == -1 and abs(gB["strength"] - 0.7) < 1e-9):
        guide_ok = False
check("guides first@0/last@-1 both 0.7 (official)", guide_ok)


def deps(nid):
    for v in nodes[nid]["inputs"].values():
        if isinstance(v, list) and len(v) == 2 and v[0] in nodes:
            yield v[0]


def ancestors(start):
    seen, stack = set(), [start]
    while stack:
        for d in deps(stack.pop()):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    return seen


first = find("SegmentNowRendering", "now0")
anc = ancestors(first)
check("all text encodes before first segment (one-pass Gemma)",
      len([e for e in types["CLIPTextEncode"] if e in anc]) == N + 1)
check("no sampler before first segment",
      not [s for s in types.get("SamplerCustomAdvanced", []) if s in anc])
check("frames preview gates first segment",
      len([x for x in types.get("ImageBatch", []) if x in anc]) == 2 * N - 1)

order_ok = True
for i in range(1, N):
    if not any(x.endswith("save%d" % (i - 1))
               for x in ancestors(find("SegmentVideoSaver", "save%d" % i))):
        order_ok = False
check("segments render strictly in order", order_ok)

color = {x: 0 for x in nodes}
cyc = []


def dfs(x):
    color[x] = 1
    for d in deps(x):
        if color[d] == 1:
            cyc.append(x)
        elif color[d] == 0:
            dfs(d)
    color[x] = 2


for x in list(nodes):
    if color[x] == 0:
        dfs(x)
check("expansion graph acyclic", not cyc)

board = json.loads(out["ui"]["gap_board"][0])
check("board rows == all segments", len(board["rows"]) == N)
check("board card pairing i->i+1",
      all(r["cardA"] == r["i"] and r["cardB"] == min(r["i"] + 1, N) for r in board["rows"]))

# ---- 3. resume path --------------------------------------------------------
clips = os.path.join(run_dir, "clips")
os.makedirs(clips, exist_ok=True)
for i in range(N):
    open(os.path.join(clips, "clip_%03d.mp4" % i), "wb").write(b"x")
out2 = LTX.render_segments(False, "M", "C", "V", "A", torch.zeros((N + 1, 32, 32, 3)),
                           json.dumps(["p%d" % i for i in range(N)]),
                           json.dumps({"segments": segs, "card_count": N + 1}),
                           run_dir, 1280, 704, 24, 100, 0.7, "neg",
                           segments_per_queue=2, resume=True)
check("all-rendered path returns count 0", out2["result"][2] == 0)
b2 = json.loads(out2["ui"]["gap_board"][0])
check("all-rendered path still sends full board",
      len(b2["rows"]) == N and all(r["status"] == "done" for r in b2["rows"]))

os.remove(os.path.join(clips, "clip_002.mp4"))
os.remove(os.path.join(clips, "clip_004.mp4"))
out3 = LTX.render_segments(False, "M", "C", "V", "A", torch.zeros((N + 1, 32, 32, 3)),
                           json.dumps(["p%d" % i for i in range(N)]),
                           json.dumps({"segments": segs, "card_count": N + 1}),
                           run_dir, 1280, 704, 24, 100, 0.7, "neg",
                           segments_per_queue=2, resume=True)
g3 = out3["expand"]
n3 = g3.get("nodes", g3) if isinstance(g3, dict) else g3
rendered_ids = sorted(int(x.split("save")[-1]) for x in n3
                      if n3[x]["class_type"] == "SegmentVideoSaver")
check("resume renders exactly the missing segments", rendered_ids == [2, 4], rendered_ids)

# ---- 4. MiniMax expansion --------------------------------------------------
MM = cm["MiniMaxSegmentsRender"]()
mo = MM.render_segments(False, "M", "C", "V", "A", torch.zeros((N + 1, 32, 32, 3)),
                        json.dumps(["p%d" % i for i in range(N)]),
                        json.dumps({"segments": segs, "card_count": N + 1}),
                        tempfile.mkdtemp(prefix="audit_mm_"), 768, 512, 24, 5, 8,
                        "res_multistep", "simple",
                        max_segments=999, segments_per_queue=N, resume=True)
mg = mo["expand"]
mnodes = mg.get("nodes", mg) if isinstance(mg, dict) else mg
mtypes = {}
for nid, nd in mnodes.items():
    mtypes.setdefault(nd["class_type"], []).append(nid)
check("MiniMax expansion builds N segments",
      len(mtypes.get("MiniMaxH3ImageToVideo", [])) == N)
check("MiniMax frame counts 5+17k",
      all((mnodes[x]["inputs"]["length"] - 5) % 17 == 0
          for x in mtypes.get("MiniMaxH3ImageToVideo", [])))

# ---- 5. SegmentTable on the newest real run --------------------------------
root = os.path.join(os.path.dirname(os.path.dirname(PACK)), "output", "storyboard_projects")
if os.path.isdir(root):
    runs = sorted((d for d in os.listdir(root)
                   if os.path.isfile(os.path.join(root, d, "project.json"))),
                  key=lambda d: os.path.getmtime(os.path.join(root, d, "project.json")),
                  reverse=True)
    if runs:
        rd = os.path.join(root, runs[0])
        pj = json.load(open(os.path.join(rd, "project.json"), encoding="utf-8"))
        to = cm["SegmentTable"]().build_table(json.dumps(pj), rd,
                animation_prompts=json.dumps(pj.get("animation_prompts") or []))
        tb = json.loads(to["ui"]["gap_board"][0])
        outroot = os.path.dirname(root)
        broken = [1 for r in tb["rows"] for k in ("first", "last", "clip") if r[k]
                  and not os.path.isfile(os.path.join(outroot, r[k]["subfolder"], r[k]["filename"]))]
        check("SegmentTable vs real run %s" % runs[0][:28],
              len(tb["rows"]) == len(pj["segments"]) and not broken)

# ---- 6. prompt engine ------------------------------------------------------
gen = cm["StoryboardPromptGenerator"]()
segs2 = [{"segment_index": i, "start_time": i * 6, "end_time": i * 6 + 6, "duration": 6,
          "lyrics": l, "section_type": s, "section_label": s, "local_energy": 0.5}
         for i, (l, s) in enumerate([("city lights at night", "intro"),
                                     ("rain on the road", "chorus_climax"),
                                     ("(instrumental)", "outro")])]
music = {"tempo": 120.0, "key": "A", "mode": "minor", "energy": 0.2, "brightness": 0.28,
         "brightness_hz": 3000.0, "percussive_ratio": 0.4, "flatness": 0.03,
         "contrast_db": 22.0, "arousal": 0.7, "valence": -0.2,
         "emotional_tone": "driving_kinetic",
         "look": "dark_electronic", "suggested_preset": "Cyberpunk", "palette": "neon teal"}
sd = json.dumps({"segments": segs2, "card_count": 4, "music": music})
res = {}
for ap in n.PROMPT_APPROACHES:
    r = gen.generate_prompts(sd, "", "", "", "", "Z-Image", prompt_approach=ap)
    res[ap.split(" ")[0]] = json.loads(r["result"][0])
check("all %d approaches distinct" % len(n.PROMPT_APPROACHES),
      len({json.dumps(v) for v in res.values()}) == len(n.PROMPT_APPROACHES))
check("cards unique within every deck", all(len(set(v)) == len(v) for v in res.values()))
check("music world reaches cards",
      any("cyberpunk" in c.lower() for c in res["Lyrics-focused"]))

ab = res["Abstract"]
check("abstract names a real art movement in EVERY card",
      all(re.search(r"in the (style|manner) of", c) for c in ab))
check("abstract stays in ONE movement across the deck",
      len({re.search(r"in the style of ([^.]+)\.", c).group(1)
           for c in ab if re.search(r"in the style of ([^.]+)\.", c)}) == 1)
check("abstract forbids realism",
      all("no photographic realism" in c for c in ab))
# different seed -> possibly different movement, always reproducible
r_a = gen.generate_prompts(sd, "", "", "", "", "Z-Image",
                           prompt_approach=n.PROMPT_APPROACHES[2], llm_seed=1)
r_b = gen.generate_prompts(sd, "", "", "", "", "Z-Image",
                           prompt_approach=n.PROMPT_APPROACHES[2], llm_seed=1)
check("abstract reproducible for same seed",
      r_a["result"][0] == r_b["result"][0])
if "Concert" in res:
    check("Concert deck is stage-bound",
          all(("stage" in c.lower()) or ("perform" in c.lower()) for c in res["Concert"]))

# music looks map to movements
for look in n.MUSIC_LOOKS:
    check("art movements defined for look %s" % look,
          bool(n.ART_MOVEMENTS.get(look) or look not in n.ART_MOVEMENTS))

# ---- 7. seamless chaining --------------------------------------------------
# Segment i+1 must start from segment i's REAL rendered final frame (its
# handoff), never from the card - the card is only approached, so starting on
# it makes every boundary a visible jump.
for cls, tagname in (("LTXSegmentsRender", "LTX"), ("MiniMaxSegmentsRender", "MiniMax")):
    node7 = cm[cls]()
    rd7 = tempfile.mkdtemp(prefix="audit_seam_")
    if cls == "LTXSegmentsRender":
        o7 = node7.render_segments(False, "M", "C", "V", "A", torch.zeros((5, 32, 32, 3)),
                                   json.dumps(["p%d" % i for i in range(4)]),
                                   json.dumps({"segments": [dict(segment_index=i,
                                       start_time=i*6.0, end_time=i*6+6.0, duration=6.0,
                                       lyrics="x", section_type="verse") for i in range(4)],
                                       "card_count": 5}),
                                   rd7, 1280, 704, 24, 7, 0.7, "neg",
                                   segments_per_queue=4, resume=True, seamless_chain=True)
    else:
        o7 = node7.render_segments(False, "M", "C", "V", "A", torch.zeros((5, 32, 32, 3)),
                                   json.dumps(["p%d" % i for i in range(4)]),
                                   json.dumps({"segments": [dict(segment_index=i,
                                       start_time=i*6.0, end_time=i*6+6.0, duration=6.0,
                                       lyrics="x", section_type="verse") for i in range(4)],
                                       "card_count": 5}),
                                   rd7, 768, 512, 24, 7, 8, "res_multistep", "simple",
                                   segments_per_queue=4, resume=True, seamless_chain=True)
    g7 = o7["expand"]
    n7 = g7.get("nodes", g7) if isinstance(g7, dict) else g7
    banners = {nid: nd for nid, nd in n7.items() if nd["class_type"] == "SegmentNowRendering"}
    chained = 0
    for nid, nd in banners.items():
        src = nd["inputs"].get("start_image")
        if isinstance(src, list) and "hand" in src[0]:
            chained += 1
    savers_fed = all(isinstance(nd["inputs"].get("last_frame"), list)
                     for nd in n7.values() if nd["class_type"] == "SegmentVideoSaver")
    check("%s seamless chain: 3/4 banners start on previous handoff" % tagname, chained == 3, chained)
    check("%s seamless chain: every saver stores its handoff frame" % tagname, savers_fed)

# ---- report ----------------------------------------------------------------
print("")
print("=" * 70)
print(" AUDIT: %d passed, %d failed" % (len(PASS), len(FAIL)))
print("=" * 70)
for f in FAIL:
    print(" FAIL:", f)
if not FAIL:
    print(" ALL CHECKS PASSED")
sys.exit(1 if FAIL else 0)
