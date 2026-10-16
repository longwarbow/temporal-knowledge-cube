# -*- coding: utf-8 -*-
"""
E14 / W3b 前置探针：在 PHEME 里筛出「适合做能力矩阵 + 辟谣注入对照」的线程。

筛选条件（用真值，不靠直觉）：
- 该线程的源贴有 veracity 标签（label ∈ {0,1,2}）
- 线程内回复有 stance 真值（stance_dict），且含 >=2 条 disagreed(3)
- 线程有文本的贴数在 [15, 80]（足够组树、又能在预算内跑完）
- 尽量涵盖多个事件（用源贴文本做事件关键词判定）

只读原数据，输出 JSON 候选清单。
"""
import json, pickle
from pathlib import Path
from collections import Counter

PHEME = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/数据/PHEME")
OUT = Path("D:/WSPforWB/10.keyan/场景1-海量短文本/实验/E14_Baseline2_EraRAG/W3b_线程候选.json")

SNOWFLAKE_EPOCH_MS = 1288834974657
STANCE = {0: "agreed", 1: "comment", 2: "appeal", 3: "disagreed"}

EVENT_KEYS = {
    "ferguson": ["ferguson", "michael brown", "#ferguson", "st. louis", "st louis", "missouri"],
    "ottawashooting": ["ottawa", "#ottawa", "parliament hill", "shooting"],
    "sydneysiege": ["sydney", "#sydneysiege", "lindt", "martin place"],
    "germanwings-crash": ["germanwings", "#germanwings", "4u9525", "airbus"],
    "charliehebdo": ["charlie hebdo", "#charliehebdo", "charliehebdo", "je suis charlie"],
}


def sf2ms(sid):
    return (int(sid) >> 22) + SNOWFLAKE_EPOCH_MS


def guess_event(text):
    t = (text or "").lower()
    for ev, kws in EVENT_KEYS.items():
        for k in kws:
            if k in t:
                return ev
    return None


def main():
    cascades = pickle.load(open(PHEME / "pheme_cascades.gpickle", "rb"))
    veracity = pickle.load(open(PHEME / "veracity_dict.pkl", "rb"))
    stance = pickle.load(open(PHEME / "stance_dict.pkl", "rb"))
    print(f"cascades={len(cascades)} veracity={len(veracity)} stance={len(stance)}")

    cands = []
    for ci, g in enumerate(cascades):
        src = None
        src_text = ""
        for nid, a in g.nodes(data=True):
            if a.get("is_source"):
                src, src_text = nid, (a.get("text") or "")
                break
        if src not in veracity:
            continue
        tws = [(nid, (a.get("text") or ""), bool(a.get("is_source")))
               for nid, a in g.nodes(data=True) if (a.get("text") or "")]
        if not (15 <= len(tws) <= 80):
            continue
        st_cnt = Counter()
        for nid, _t, is_src in tws:
            if not is_src and nid in stance:
                st_cnt[stance[nid]] += 1
        n_dis = st_cnt.get(3, 0)
        n_agr = st_cnt.get(0, 0)
        n_com = st_cnt.get(1, 0)
        if n_dis >= 2 and (n_agr + n_com) >= 3:
            ms = [sf2ms(n) for n, _t, _s in tws]
            cands.append({
                "thread_idx": ci, "source_id": src,
                "label": veracity[src], "label_name": ["true", "false", "unverified"][veracity[src]],
                "n_text": len(tws), "n_agreed": n_agr, "n_comment": n_com, "n_disagreed": n_dis,
                "event_guess": guess_event(src_text),
                "iso_min": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime(min(ms)/1000)),
                "src_text_head": src_text[:140],
            })
    cands.sort(key=lambda c: (-c["n_disagreed"], -c["n_text"]))
    OUT.write_text(json.dumps({"n_candidates": len(cands), "top": cands[:25]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[候选] 符合条件线程 = {len(cands)}，写入 {OUT.name}")
    for c in cands[:12]:
        print(f"  idx={c['thread_idx']:5d} n={c['n_text']:3d} dis={c['n_disagreed']} "
              f"agr={c['n_agreed']} lbl={c['label_name']:10s} ev={c['event_guess']} {c['iso_min']}")
        print(f"      src: {c['src_text_head']}")


if __name__ == "__main__":
    main()
