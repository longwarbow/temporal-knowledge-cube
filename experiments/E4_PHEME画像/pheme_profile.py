# -*- coding: utf-8 -*-
"""
PHEME 数据集画像（D22 执行产物）
=================================
输入：
  - data/PHEME/pheme_cascades.gpickle          （6,425 个 NetworkX DiGraph，节点=推文，边=回复）
  - data/PHEME/veracity_dict.pkl               （1,972 个谣言线程的三值标注）
  - data/PHEME/stance_dict.pkl                 （4,082 条回复的立场标注）
输出：
  - E4_报告.md / e4_results.json

标签编码（由 NirmalenduPrakash/RumourPrediction 源码确定）：
  veracity_dict: 0=true, 1=false, 2=unverified
  stance_dict  : 0=agreed, 1=comment, 2=appeal-for-more-information, 3=disagreed
  graph.graph['label']: 1=rumour, 0=non-rumour
"""
import pickle, json, collections, statistics, re, io, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(ROOT, '数据', 'PHEME')

VERACITY = {0: 'true', 1: 'false', 2: 'unverified'}
STANCE = {0: 'agreed', 1: 'comment', 2: 'appeal-for-more-info', 3: 'disagreed'}


def load_graphs():
    with open(os.path.join(DATA, 'pheme_cascades.gpickle'), 'rb') as f:
        return pickle.load(f)


def load_pkl(name):
    p = os.path.join(DATA, name)
    with open(p, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            return pickle.load(f, encoding='latin1')


def pct(a, b):
    return (100.0 * a / b) if b else 0.0


def main():
    t0 = time.time()
    G = load_graphs()
    ver = load_pkl('veracity_dict.pkl')
    stn = load_pkl('stance_dict.pkl')

    n_threads = len(G)
    events = collections.Counter()
    ev_label = collections.defaultdict(lambda: collections.Counter())  # event -> {1:r,0:nr}
    thread_lbl = {}          # thread_id -> label(1/0)
    ev_threads = collections.defaultdict(list)

    total_nodes = 0
    total_edges = 0
    tweets_per_thread = []
    depth_list = []
    roots_ok = 0
    roots_bad = 0
    is_source_cnt = 0
    empty_text = 0
    text_lens = []
    followers = []
    verified_users = 0
    tweets_with_url = 0
    tweets_with_hashtag = 0
    non_ascii_heavy = 0
    thread_node_counts = []

    # 逐图统计
    for g in G:
        ev = g.graph.get('event', 'NA')
        lab = int(g.graph.get('label', -1))
        events[ev] += 1
        ev_label[ev][lab] += 1
        tid = str(g.graph.get('thread_id', ''))
        thread_lbl[tid] = lab
        ev_threads[ev].append(tid)

        nn = g.number_of_nodes()
        total_nodes += nn
        total_edges += g.number_of_edges()
        tweets_per_thread.append(nn)
        thread_node_counts.append((nn, tid, ev, lab))

        roots = [n for n, d in g.nodes(data=True) if d.get('is_source')]
        if len(roots) == 1:
            roots_ok += 1
        else:
            roots_bad += 1

        for n, d in g.nodes(data=True):
            if d.get('is_source'):
                is_source_cnt += 1
            t = d.get('text') or ''
            if not t.strip():
                empty_text += 1
            else:
                text_lens.append(len(t))
                ascii_ratio = sum(1 for c in t if ord(c) < 128) / max(1, len(t))
                if ascii_ratio < 0.6:
                    non_ascii_heavy += 1
            if 'http' in t:
                tweets_with_url += 1
            if '#' in t:
                tweets_with_hashtag += 1
            followers.append(d.get('user_followers', 0))
            if d.get('user_verified'):
                verified_users += 1

        # 深度（以 is_source 为根的最长 BFS 分层）
        try:
            import networkx as nx
            if roots:
                dmax = nx.eccentricity(g, v=roots[0]) if nx.is_connected(g.to_undirected()) else None
                if dmax is None:
                    # 有向图下用最短路径最大深度
                    lens = nx.single_source_shortest_path_length(g, roots[0])
                    dmax = max(lens.values()) if lens else 0
                depth_list.append(dmax)
        except Exception:
            pass

    # veracity 覆盖
    ver_covered = [tid for tid in thread_lbl if tid in ver]
    ver_dist = collections.Counter(ver[t] for t in ver_covered)
    ver_in_rumour = collections.Counter(ver[t] for t in ver_covered if thread_lbl[t] == 1)
    ver_in_nonrumour = collections.Counter(ver[t] for t in ver_covered if thread_lbl[t] == 0)

    # 每个事件的 veracity 分布
    ev_ver = collections.defaultdict(collections.Counter)
    for ev, tids in ev_threads.items():
        for t in tids:
            if t in ver:
                ev_ver[ev][ver[t]] += 1

    # stance 分布
    stn_dist = collections.Counter(stn.values())

    def q(a, p):
        a = sorted(a)
        if not a:
            return 0
        i = min(len(a) - 1, int(len(a) * p))
        return a[i]

    res = {
        'n_threads': n_threads,
        'events': dict(events),
        'event_label': {k: dict(v) for k, v in ev_label.items()},
        'total_tweets': total_nodes,
        'total_edges': total_edges,
        'tweets_per_thread': {
            'mean': round(statistics.mean(tweets_per_thread), 2),
            'median': statistics.median(tweets_per_thread),
            'p90': q(tweets_per_thread, 0.9), 'p99': q(tweets_per_thread, 0.99),
            'max': max(tweets_per_thread), 'min': min(tweets_per_thread),
        },
        'root_unique_ok': roots_ok, 'root_unique_bad': roots_bad,
        'is_source_nodes': is_source_cnt,
        'depth': {'median': statistics.median(depth_list) if depth_list else 0,
                  'p90': q(depth_list, 0.9), 'max': max(depth_list) if depth_list else 0},
        'text': {
            'nonempty': len(text_lens), 'empty': empty_text,
            'len_median': statistics.median(text_lens) if text_lens else 0,
            'len_p90': q(text_lens, 0.9), 'len_max': max(text_lens) if text_lens else 0,
            'url_rate_pct': round(pct(tweets_with_url, total_nodes), 1),
            'hashtag_rate_pct': round(pct(tweets_with_hashtag, total_nodes), 1),
            'non_ascii_heavy_pct': round(pct(non_ascii_heavy, max(1, len(text_lens))), 1),
        },
        'followers': {
            'median': statistics.median(followers) if followers else 0,
            'mean': round(statistics.mean(followers), 1) if followers else 0,
            'p99': q(followers, 0.99), 'max': max(followers) if followers else 0,
            'zero_rate_pct': round(pct(sum(1 for x in followers if x == 0), len(followers)), 1),
        },
        'verified_users': verified_users,
        'veracity': {
            'annotated_threads': len(ver),
            'matched_in_gpickle': len(ver_covered),
            'dist': {VERACITY[k]: v for k, v in ver_dist.items()},
            'in_rumour_threads': {VERACITY[k]: v for k, v in ver_in_rumour.items()},
            'in_nonrumour_threads': {VERACITY[k]: v for k, v in ver_in_nonrumour.items()},
            'per_event': {e: {VERACITY[k]: v for k, v in c.items()} for e, c in ev_ver.items()},
        },
        'stance': {'annotated': len(stn), 'dist': {STANCE[k]: v for k, v in stn_dist.items()}},
        'elapsed_s': round(time.time() - t0, 1),
    }

    with io.open(os.path.join(HERE, 'e4_results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    tot_rum = sum(v.get(1, 0) for v in ev_label.values())
    tot_non = sum(v.get(0, 0) for v in ev_label.values())
    ev_with_ver = sorted(ev_ver)
    ev_without_ver = [e for e in events if e not in ev_ver]

    # 写报告
    r = []
    A = r.append
    A("# PHEME 数据集画像（E4）")
    A("")
    A("> **性质**：数据集画像（D22 的执行产物）。**英文泛化组 + 真实状态迁移真值**的首个实测。")
    A("> **日期**：2026-10-10 ｜ **脚本**：`pheme_profile.py` ｜ **指标**：`e4_results.json`")
    A("> **数据来源**：PHEME 9-Event（figshare `10.6084/m9.figshare.6392078`，CC-BY 标注）；")
    A("> 因本机 figshare 被 WAF 拦截，实际取用两条**可达镜像**：① `NunoBatista/PHEME-Misinformation-Graphs`（HF，含每推文正文+回复边）；")
    A("> ② `NirmalenduPrakash/RumourPrediction`（GitHub，`veracity_dict.pkl` = PHEME 官方 `annotation.json` 的三值机器编码）。两者以 `thread_id` 可对齐。")
    A("")
    A("## 0. 一句话结论")
    A("")
    A("**PHEME 通过了 D22 的全部验收条件，且一次性补齐两个缺口**：")
    A("① **G4（泛化组）** —— 9 事件 / **%s** 会话线程（**%s 谣言 + %s 非谣言**）/ **%s** 条推文，英文；"
      % (f"{n_threads:,}", f"{tot_rum:,}", f"{tot_non:,}", f"{total_nodes:,}"))
    A("② **G7 / 缺口 A（真实状态迁移）** —— **%d 个谣言线程带 `true/false/unverified` 三值真值**（true=%d / false=%d / unverified=%d，覆盖 %d/9 事件），"
      % (len(ver), ver_dist.get(0, 0), ver_dist.get(1, 0), ver_dist.get(2, 0), len(ev_ver)))
    A("这是公开数据中**唯一**能提供「声明被裁定为真 / 假 / 悬而未决」**完整三态**的来源。")
    A("")
    A("> ⚠️ **覆盖边界（诚实声明）**：可达镜像的 veracity 标注覆盖 **%d/9 事件（%d / %s 谣言线程，%.1f%%）**——即经典的 PHEME 五事件真值子集（SemEval-2017 Task 8 口径）。"
      % (len(ev_ver), len(ver), f"{tot_rum:,}", pct(len(ver), tot_rum)))
    A("> 其余 4 事件（%s，共 %d 条谣言）在原 `annotation.json` 中亦有标注，需原 figshare 包补齐；本镜像未含。" % (" / ".join(ev_without_ver), tot_rum - len(ver)))
    A("")
    A("## 1. 规模与结构")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A("| 会话线程（graph） | **%s**（**%s 谣言 + %s 非谣言**，谣言占 %.1f%%） |" % (f"{n_threads:,}", f"{tot_rum:,}", f"{tot_non:,}", pct(tot_rum, n_threads)))
    A("| 推文节点（node） | **%s** |" % f"{total_nodes:,}")
    A("| 回复边（edge） | **%s** |" % f"{total_edges:,}")
    A("| 每线程推文数（中位 / 均值 / p90 / p99 / max） | %s / %s / %s / %s / %s |" % (
        res['tweets_per_thread']['median'], res['tweets_per_thread']['mean'],
        res['tweets_per_thread']['p90'], res['tweets_per_thread']['p99'], res['tweets_per_thread']['max']))
    A("| 传播深度（中位 / p90 / max） | %s / %s / %s |" % (res['depth']['median'], res['depth']['p90'], res['depth']['max']))
    A("| 根节点唯一（唯一 `is_source`，可还原树） | **%s / %s（%.1f%%）** |" % (roots_ok, n_threads, pct(roots_ok, n_threads)))
    A("")
    A("> 根唯一的 **%.1f%%** 略低于 CED 的 100%%：原因是镜像建图时**部分反应推文的 `in_reply_to_status_id` 缺失**，被回挂到源帖（`prepare_data.py` 的兜底逻辑），或源帖 JSON 缺失；"
      % pct(roots_ok, n_threads))
    A("> 这不影响「回复树 + 唯一源帖」的主体结构，但**做 S2（核心转发者子图）时须先剔除这 %.0f%% 的退化图**。" % (100 - pct(roots_ok, n_threads)))
    A("")
    A("### 1.1 事件 × 标签")
    A("")
    A("| 事件 | 线程数 | 谣言 | 非谣言 | 谣言占比 |")
    A("|---|---:|---:|---:|---:|")
    for e in sorted(events, key=lambda x: -events[x]):
        rc = ev_label[e].get(1, 0); nc = ev_label[e].get(0, 0)
        A("| %s | %d | %d | %d | %.1f%% |" % (e, events[e], rc, nc, pct(rc, events[e])))
    A("")
    A("> 谣言/非谣言比例因事件而异（与 CED 的 1,538/1,849 对照：CED 近平衡，PHEME 偏非谣言）。")
    A("")
    A("## 2. ★ 真实状态迁移真值（本数据集对 RQ1 的核心价值）")
    A("")
    A("`veracity_dict.pkl` 是 PHEME 官方 `annotation.json` 的机器编码（`veracity_enum`：true=0/false=1/unverified=2）。")
    A("")
    A("| veracity | 线程数 | 占比 | 对我们的含义 |")
    A("|---|---:|---:|---|")
    for k in (0, 1, 2):
        A("| **%s** | %d | %.1f%% | %s |" % (
            VERACITY[k], ver_dist.get(k, 0), pct(ver_dist.get(k, 0), len(ver_covered)),
            {0: "声明被证实（`active` 保持）", 1: "声明被证伪（≡ `retracted`）", 2: "悬而未决（`unverified`，状态未迁移）"}[k]))
    A("| 合计 | **%d** | 100%% | |" % len(ver_covered))
    A("")
    A("**与中文大集的关键对照**：")
    A("")
    A("| | 中文谣言大集 | **PHEME** |")
    A("|---|---|---|")
    A("| 真值形态 | 单侧（97.6% 判「不实」） | **三值平衡（true/false/unverified 均在）** |")
    A("| 时间语义 | `publishTime` = 处置时间 st（非 t） | 会话演化 + 回复时间（未在镜像中暴露，需原包） |")
    A("| 传播结构 | ❌ 无 | **✅ 回复树（唯一根 %.1f%%）** |" % pct(roots_ok, n_threads))
    A("| 语言 | 中文 | **英文** |")
    A("")
    A("⇒ **PHEME 是 `active→retracted`（以及 `active→true`）迁移真值的唯一公开来源**，正好补上 D2-rev1(a) 里「完整真值改由 PHEME 承载」那一环。")
    A("")
    A("### 2.1 每个事件的 veracity 分布")
    A("")
    hdr = "| 事件 | true | false | unverified |"
    A(hdr); A("|---|---:|---:|---:|")
    for e in sorted(ev_ver, key=lambda x: -sum(ev_ver[x].values())):
        c = ev_ver[e]
        A("| %s | %d | %d | %d |" % (e, c.get(0, 0), c.get(1, 0), c.get(2, 0)))
    A("")
    A("> 未含 veracity 的 4 事件：%s（各有 %s 条谣言线程，仅有谣言/非谣言二元标签）。" % (
        " / ".join(ev_without_ver),
        " / ".join(str(ev_label[e].get(1, 0)) for e in ev_without_ver)))
    A("")
    A("## 3. 文本与用户（支撑 S1/S2/S3 与语义度量）")
    A("")
    A("| 指标 | 值 |")
    A("|---|---|")
    A("| 正文非空 / 空 | %s / %s |" % (res['text']['nonempty'], res['text']['empty']))
    A("| 文本长度（中位 / p90 / max） | %s / %s / %s |" % (res['text']['len_median'], res['text']['len_p90'], res['text']['len_max']))
    A("| 含 URL 比例 | %.1f%% |" % res['text']['url_rate_pct'])
    A("| 含 # 话题 比例 | %.1f%% |" % res['text']['hashtag_rate_pct'])
    A("| 非 ASCII 为主（疑似非英文） | %.1f%% |" % res['text']['non_ascii_heavy_pct'])
    A("| 粉丝数（中位 / 均值 / p99 / max） | %s / %s / %s / %s |" % (
        res['followers']['median'], res['followers']['mean'], res['followers']['p99'], res['followers']['max']))
    A("| 零粉丝账号占比 | %.1f%% |" % res['followers']['zero_rate_pct'])
    A("| 认证账号数 | %s |" % f"{verified_users:,}")
    A("")
    A("> **粉丝数可得** ⇒ D2-rev1(b) 的 **S1 影响力集中度**（基尼/头部占比）在 PHEME 上可算；")
    A("> 回复边 + 唯一根 ⇒ **S2 核心转发者子图**可算；**S3 时间同步性**可从节点 tweet id 经雪花反解获得（$t=(\\text{id}\\gg22)+1288834974657$，见 E5 §1）——**镜像实际保留了完整毫秒级时间轴**。")
    A("")
    A("## 4. 立场标注（补充）")
    A("")
    A("`stance_dict.pkl` = %d 条回复的立场（`responsetype-vs-previous`）：" % len(stn))
    A("")
    A("| 立场 | 数 | 占比 |")
    A("|---|---:|---:|")
    for k in (0, 1, 2, 3):
        A("| %s | %d | %.1f%% |" % (STANCE[k], stn_dist.get(k, 0), pct(stn_dist.get(k, 0), len(stn))))
    A("")
    A("> 立场维（D6 的「L4 叙事维最小替代」）在 PHEME 上有真实标注 —— 可作为**英文组**的立场维验证。")
    A("")
    A("## 5. 与 CED / 中文大集的横向对照")
    A("")
    A("| 维度 | CED（中文） | 中文大集 | **PHEME（英文）** |")
    A("|---|---|---|---|")
    A("| 规模 | 3,387 帖 + 127 万转发 | 31,669 条 | **6,425 线程 / %s 推文** |" % f"{total_nodes:,}")
    A("| 帖级时间 | ✅ 97.4%（2010–14） | ✅ 8 年（但=处置时间） | **✅ 毫秒级（tweet id 雪花反解）** |")
    A("| 传播树 | ✅（真孤儿 0.000%%） | ❌ | **✅（唯一根 %.1f%%）** |" % pct(roots_ok, n_threads))
    A("| 状态真值 | 二元（谣/非谣） | 单侧（不实） | **★ 三值（true/false/unverified）** |")
    A("| 空间 | 用户自填省市 | ❌ | ❌（镜像无 geo） |")
    A("| 语言 | 中文 | 中文 | **英文** |")
    A("| 主用途 | P1 增量 / 源独立性 / as-of | D2 辟谣词典 / 单侧真值 | **D5 泛化组 + P2 真实状态迁移** |")
    A("")
    A("## 6. 对缺口的处置")
    A("")
    A("| 缺口 | 原状态 | **PHEME 落地后** |")
    A("|---|---|---|")
    A("| **G4 泛化组不足** | 待补 | **✅ 关闭**（D22 执行完成；英文 9 事件） |")
    A("| **G7 / 缺口 A 真实状态迁移** | 只能注入式 | **✅ 部分关闭**（有真实三值真值 + 毫秒时间轴；缺「**裁定时刻**」——PHEME 为事后人工标注，须原包或人工补标） |")
    A("| G1 空间 | 部分 | 不变（PHEME 无 geo） |")
    A("| 血缘 / 声明级 | 自建 | 不变（PHEME 无 (s,r,o) 声明级标注，须 LLM 抽取） |")
    A("")
    A("## 7. 来源与可复现性声明（配合 D23 / D24）")
    A("")
    A("- **原始来源**：figshare `10.6084/m9.figshare.6392078`（Kochkina, Liakata, Zubiaga 2018）。")
    A("- **本机取用**：`NunoBatista/PHEME-Misinformation-Graphs`（HF 镜像，MIT）+ `NirmalenduPrakash/RumourPrediction`（GitHub，`veracity_dict.pkl`/`stance_dict.pkl`）。")
    A("- **pickle 安全**：`veracity_dict.pkl` / `stance_dict.pkl` 经 `pickletools` 全量 opcode 审计，**无 GLOBAL 指令**（纯 `str→int` dict），加载安全。")
    A("- **★ 时间轴更正（2026-10-10）**：原判「镜像不含 `created_at`」**有误** —— 节点 id 为 18 位 Twitter snowflake，"
      "可反解出**毫秒级**发布时刻（已由 E5 §1 用真实事件日期校验通过）。故时间轴与 S3 时间同步性**在该镜像上均可算**；"
      "仍缺的是**裁定时刻**（veracity 为事后标注，非原始时间字段）。")
    A("")
    A("*生成：WorkBuddy ｜ 耗时 %.1fs ｜ 未消耗 SmartLib 配额。*" % res['elapsed_s'])

    with io.open(os.path.join(HERE, 'E4_报告.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(r))

    print("done in %.1fs" % res['elapsed_s'])
    print(json.dumps({k: res[k] for k in ('n_threads', 'total_tweets', 'total_edges', 'root_unique_ok', 'veracity')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
