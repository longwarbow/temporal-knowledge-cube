# -*- coding: utf-8 -*-
"""
E15 / Phase 1 —— L2 话题侧：线程 → 锚定声明抽取（含 A4 谓词族复用）
-----------------------------------------------------------------------------
单位：PHEME 一条**线程**（= 一个源贴所主张的叙事）。取其源贴文本（+可选高 pagerank 回复）
      抽出一个**锚定声明**，作为该线程在话题维上的坐标。

复用：E13 判据 §2.5 的 **A4 封闭族表（10 族）** 作为谓词族词表（已冻结、已验证）。
      族 = 话题槽位的"谓词"维度；subject/aspect = "主语/侧面"维度。

输出（JSONL，可断点续跑）：E15_P1_声明_{event}.jsonl
  {thread_id, source_id, event, day, ms, label(veracity), n_nodes, n_text,
   subject, aspect, family, value, polarity, claim, salience, conf}

用法：
  python e15_p1_extract.py --event sydneysiege-all-rnr-threads            # 全量（含未标注）
  python e15_p1_extract.py --event sydneysiege-all-rnr-threads --labeled-only
  python e15_p1_extract.py --event sydneysiege-all-rnr-threads --limit 50 --workers 6
"""
import argparse
import collections
import datetime
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))          # 10.keyan
DATA = os.path.join(ROOT, "场景1-海量短文本", "数据", "PHEME")
sys.path.insert(0, os.path.join(ROOT, "场景1-海量短文本", "实验", "E14_Baseline2_EraRAG"))
from e14_adapter import snowflake_to_ms, ms_to_iso                    # noqa: E402

FAMILIES = {
    "FATALITY": "killed/died/dead/shot dead/found dead/fatalities/toll/body",
    "INJURY": "injured/wounded/hurt/hospitalized/critical/stable condition/paramedics",
    "AMBIG_PERSON": "victim(s)/casualty(casualties)/affected",
    "CUSTODY_HELD": "hostage(s)/held hostage/taken hostage/hostage situation/kidnapped/abducted/seized/detained",
    "CUSTODY_EXITED": "escaped/fled/released/freed/rescued/have left",
    "PERPETRATOR_EXISTS": "gunman/shooter/suspect/attacker/terrorist/assailant/killer/bomber",
    "PERPETRATOR_IDENTITY": "named/identified/background/motive/profile/convert/refugee/offender/record",
    "FORCE_SHOTS": "gunfire/shots/shooting/shootout/opened fire/burst in",
    "FORCE_EXPLOSION": "explosion/blast/bomb/detonated/grenade",
    "POLICE_OPERATION": "standoff/siege/lockdown/surrounded/holed up/stormed/raid/evacuated",
}
FAM_LIST = "\n".join("- %s : %s" % (k, v) for k, v in FAMILIES.items())

SYS = ("You extract the single ANCHORED CLAIM asserted by a social-media post during a "
       "crisis event, for a topic-dimension annotation in a rumour-analysis benchmark. "
       "You are literal: you report what the post asserts, not what you believe.")

PROMPT = """Crisis event: {event}
Date: {day}
Post (source of the thread):
\"\"\"{text}\"\"\"

Task: state the ONE anchored claim this post asserts — the fact whose spread defines this
thread's narrative.

1. subject  : 2-5 words, the incident/object the claim is about
              (e.g. "lindt cafe siege", "ottawa parliament shooting", "germanwings crash").
2. aspect   : 2-4 words, WHICH PROPERTY is asserted — the angle of this post
              (e.g. "hostage count", "gunman identity", "police operation", "death toll",
               "islamic flag", "public reaction"). Use the wording that best fits the post.
3. family   : the predicate family, chosen ONLY from this closed list, else NONE:
{families}
4. value    : the asserted value/state in a few words (a number if the post gives one), else "-".
5. polarity : "assert" if the post affirms the claim, "deny" if it denies/refutes one.
6. salience : 1-5, how central this claim is to the post (5 = it is the point of the post).
7. claim    : one short sentence normalising the assertion (no URLs, no @handles).

Rules: no commentary inside the fields; if the post asserts nothing factual (pure emotion,
question, or retweet boilerplate only), set family="NONE", value="-", salience<=2.

Answer ONLY with compact JSON:
{{"subject":"...","aspect":"...","family":"...","value":"...","polarity":"assert|deny","salience":1-5,"claim":"..."}}"""


def load_env():
    env = {}
    for line in open(os.path.join(ROOT, ".env_e14"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def norm_text(t, n=700):
    t = re.sub(r"https?://\S+", " ", t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:n]


def extract(client, model, event, day, text, retries=3):
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": PROMPT.format(
               event=event, day=day, text=norm_text(text), families=FAM_LIST)}]
    for i in range(retries):
        try:
            r = client.chat.completions.create(model=model, messages=msg,
                                               temperature=0.0, max_tokens=260)
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                o = json.loads(m.group(0))
                o["family"] = str(o.get("family", "NONE")).strip().upper().replace(" ", "_")
                if o["family"] not in FAMILIES:
                    o["family"] = "NONE"
                return o
        except Exception as e:                                        # noqa: BLE001
            if i == retries - 1:
                return {"family": "ERR", "err": f"{type(e).__name__}: {e}"[:120]}
            time.sleep(1.5 * (i + 1))
    return {"family": "PARSE_FAIL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--labeled-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-text", type=int, default=15)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    env = load_env()
    model = env.get("E14_LLM_MODEL", "glm-4-flash")
    from openai import OpenAI
    client = OpenAI(base_url=env["E14_LLM_BASE_URL"], api_key=env["E14_LLM_API_KEY"], timeout=90.0)

    G = pickle.load(open(os.path.join(DATA, "pheme_cascades.gpickle"), "rb"))
    ver = pickle.load(open(os.path.join(DATA, "veracity_dict.pkl"), "rb"))

    jobs = []
    for gi, g in enumerate(G):
        if g.graph.get("event") != args.event:
            continue
        src, stext = None, ""
        for nid, a in g.nodes(data=True):
            if a.get("is_source"):
                src = str(nid)
                stext = a.get("text") or ""
                break
        if src is None:
            continue
        label = ver.get(src)
        if args.labeled_only and label is None:
            continue
        if len(norm_text(stext)) < args.min_text:
            continue
        tids = [str(n) for n, a in g.nodes(data=True) if (a.get("text") or "").strip()]
        ms = min(snowflake_to_ms(t) for t in tids)
        day = datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")
        jobs.append({"thread_idx": gi, "source_id": src, "event": args.event, "day": day,
                     "ms": ms, "iso": ms_to_iso(ms), "label": label,
                     "n_nodes": g.number_of_nodes(), "n_text": len(tids), "text": stext})
    jobs.sort(key=lambda j: j["ms"])
    if args.limit:
        jobs = jobs[:args.limit]
    print("事件 %s：待抽取线程 %d（labeled_only=%s）" % (args.event, len(jobs), args.labeled_only))

    out = args.out or os.path.join(BASE, "E15_P1_声明_%s.jsonl" % args.event.split("-")[0])
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["source_id"])
            except Exception:                                          # noqa: BLE001
                pass
        print("  断点续跑：已完成 %d" % len(done))
    todo = [j for j in jobs if j["source_id"] not in done]

    t0 = time.time()
    fout = open(out, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract, client, model, j["event"], j["day"], j["text"]): j
                for j in todo}
        n = 0
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            j = futs[fut]
            o = fut.result()
            rec = {k: j[k] for k in ("source_id", "event", "day", "ms", "iso", "label",
                                     "n_nodes", "n_text")}
            rec.update({k: o.get(k) for k in ("subject", "aspect", "family", "value",
                                              "polarity", "salience", "claim")})
            rec["text"] = norm_text(j["text"], 320)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                fout.flush()
                print("  %d/%d  %.1f min" % (n, len(todo), (time.time() - t0) / 60.0), flush=True)
    fout.close()
    print("完成：新增 %d，总 %d 条 → %s（%.1f min）" %
          (len(todo), len(done) + len(todo), out, (time.time() - t0) / 60.0))

    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    print("\n族分布：", dict(collections.Counter(r["family"] for r in rows).most_common()))
    print("日分布：", dict(sorted(collections.Counter(r["day"] for r in rows).items())))
    print("aspect top12：", collections.Counter(r["aspect"] for r in rows).most_common(12))


if __name__ == "__main__":
    main()
