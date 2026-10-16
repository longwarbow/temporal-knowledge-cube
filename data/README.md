# Data manifest

**No third-party dataset is redistributed in this repository.** This file states
where each dataset comes from, its licence, and — where it matters — the
snapshot date, because several of these sources are live and will not reproduce
byte-for-byte on a later date.

---

## 1. Datasets

| Dataset | Role in the paper | Canonical source | Licence | Snapshot used |
|---|---|---|---|---|
| **CED** (Chinese Event Dataset) | repost / cascade stream; 3,387 source posts and 1,278,493 forwarding events; the physical-cost carrier (E16) and the propagation-tree carrier (E2, E11) | public GitHub archive (codeload tarball of the released repository) | released for research use | fixed tarball; retrieval script pinned |
| **CTAmap** | versioned geographic dimension: 450 real administrative-division changes, 2000–2025, plus 81 county/city/province vector sets | Zenodo, DOI `10.5281/zenodo.22179207` | **CC-BY-4.0** | 2026-10 release |
| **PHEME** | claim-state ground truth: 6,425 threads, 105,747 tweets, 9 events; veracity labels for 5 of the 9 events | Hugging Face mirror; the annotation files also via jsDelivr | released for research use | the mirror revision fetched during the run |
| **Wikidata** | declaration-level retraction truth (E8, E9): 10,970,321 `rank=deprecated` statements | public SPARQL endpoint (WDQS), `query.wikidata.org` | **CC0** | **2026-10-11** |
| **Weibo21** | domain gold standard, 9,592 posts | public release | released for research use | as released |
| **WeiboHAN** | cross-check corpus, 6,494 posts | public release | released for research use | as released |
| **Chinese large corpus** | single-sided ground truth, 31,669 posts (97.6 % labelled "not factual") | public release | released for research use | as released |

Time-of-publication data for PHEME comes from the Twitter snowflake ids decoded
locally, not from a scraped `created_at` field, so the timeline is reproducible
from the released ids alone.

---

## 2. Snapshot caveat (important)

**Wikidata is a live graph.** During profiling, the whole-graph count of
`rank=deprecated` statements moved by four statements within twenty minutes.
Every Wikidata number in the paper must be read together with its query date,
which is stamped in the artefacts; the counts reported in §7.6 of the paper
describe the graph **as of 2026-10-11**. Re-running those queries today will
give a slightly different count — that is expected, and it is not a failure of
the reproduction.

---

## 3. Retrieval policy

All retrieval is scripted and channel-pinned. Where a platform's TLS stack was
intercepted on our hosts, the scripts shell out to `curl` with an explicit
`--noproxy '*'`, retry under a documented policy, and send a `User-Agent`
carrying a contact address so that the remote operator can identify the traffic.

**No proprietary API is a dependency of any reported number.** The one API used
(E14's LLM endpoint) is free-tier and the experiment is designed so that no
absolute token or wall-clock figure is claimed — every cost is a ratio to a full
rebuild measured on the same host in the same run.

---

## 4. Reproducing the data layer

1. Read the dataset's row above and the manifest inside the relevant
   `experiments/` directory; each experiment names the exact files it consumes.
2. Fetch from the canonical source. Do not fetch from a mirror unless the row
   says the mirror is the pinned channel (PHEME).
3. For Wikidata, record the query date alongside any output you intend to
   compare against the paper.
