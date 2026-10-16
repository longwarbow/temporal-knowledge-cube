# -*- coding: utf-8 -*-
"""生成 supplementary 的浮动体分片（9 个），按补充材料编号顺序落盘到 floats/。"""
import io, os, re

SRC = '../build/_m1.tex'
OUTDIR = 'floats'

# 顺序 = 在补充材料中出现的顺序；(表/图计数器各自独立递增)
ORDER = [
    ('tab:data',         'S1 · Dataset table',                              r'\S7.1 (Setup)'),
    ('tab:independence', 'S2 · Source-independence features (E3)',          r'\S7.4.3'),
    ('tab:alert',        'S3 · $Q_3$ alert criterion, out of sample (E11)', r'\S7.4.3'),
    ('__crosscheck__',   'S4 · The seven independent cross-checks (E7)',    r'\S7.5'),
    ('__e9__',           'S5 · Adjudication time from Wikidata history (E9)', r'\S7.6'),
    ('tab:baselines',    'S6 · Baselines',                                  r'\S7.7'),
    ('fig:axes',         'Figure S1 · Three axes, one shape',               r'\S7.2--\S7.4'),
    ('fig:invalid',      'Figure S2 · Real retraction over the PHEME false set (E5)', r'\S7.4.1'),
    ('fig:threshold',    'Figure S3 · Calibrating the sparsity threshold (E12)', r'\S7.8'),
]

HAND = {}

HAND['__crosscheck__'] = r"""\begin{table}[htbp]
\caption{The seven independent cross-checks attempted in the end-to-end
pipeline (E7): quantities computed by two different mechanisms, months apart.
Six reproduce bit-for-bit; the seventh is a cost ratio that agrees to the same
order under a different cell definition.}
\label{stab:crosscheck}
\centering
\small
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}p{0.30\textwidth}p{0.26\textwidth}p{0.38\textwidth}@{}}
\toprule
\textbf{Cross-check} & \textbf{Independently measured} & \textbf{Reproduced in the pipeline}\\
\midrule
E1 equivalence (13 checkpoints) & 0 differences & 0 differences\\
E1 cost ratio & 5.6$\times$ & 7.0$\times$ (same order; different cell definition)\\
E2 city-level $\rho$ & 0.0045 & \textbf{0.0045, digit-for-digit} (hit the same 4 tokens)\\
E2 province-level $\rho$ & 0 & 0\\
E5 $|\mathrm{Affect}|$ (hour) & 224 & 224\\
E6 fully invalid cells & 28 & 28\\
Baseline \#4 maximum bias & 23.9\% (germanwings) & 23.9\% (germanwings)\\
\bottomrule
\end{tabular}
\end{table}"""

HAND['__e9__'] = r"""\begin{table}[htbp]
\caption{Adjudication time recovered from the wiki revision history (E9):
sampling 62 deprecated statements across six properties, locating the first
revision in which the target statement is deprecated and verifying that the
predecessor revision is not.}
\label{stab:e9}
\centering
\small
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}p{0.40\textwidth}p{0.52\textwidth}@{}}
\toprule
\textbf{Quantity} & \textbf{Result}\\
\midrule
adjudication timestamp recovered & 60 / 62 (96.8\%)\\
lifetime (flip $-$ birth) available & 59 / 62 (95.2\%)\\
adjudication year & 2016--2026; \textbf{98.3\% $\ge$ 2017} (mode: 2025, then 2024)\\
lifetime, median / p90 / max & \textbf{766\,d $\approx$ 2.10\,y} / 3,451\,d (9.45\,y) / 4,330\,d (11.86\,y)\\
lifetime $\le$ 1 day & \textbf{13 / 59 (22.0\%)}\\
lifetime $>$ 1 year & 35 / 59 (59.3\%)\\
cost & median 9 API calls per statement (mean 14.2; 851 total)\\
\bottomrule
\end{tabular}
\end{table}"""

s = io.open(SRC, encoding='utf-8').read()
ENV = re.compile(r'\\begin\{(table\*?|figure\*?)\}(.*?)\\end\{\1\}', re.S)
blocks = {}
for m in ENV.finditer(s):
    for key, _t, _p in ORDER:
        if key.startswith('__'):
            continue
        if '\\label{%s}' % key in m.group(0):
            blocks[key] = m.group(0)
missing = [k for k, _t, _p in ORDER if not k.startswith('__') and k not in blocks]
if missing:
    raise SystemExit('未找到浮动体: %s' % missing)

if not os.path.isdir(OUTDIR):
    os.makedirs(OUTDIR)

for i, (key, title, ptr) in enumerate(ORDER, 1):
    if key.startswith('__'):
        body = HAND[key]
    else:
        body = blocks[key].replace('\\label{%s}' % key, '\\label{s%s}' % key)
        body = body.replace('[!t]', '[htbp]')
    # 跨文档指向主论文的节
    body = re.sub(r'\\Sref\{([^}]*)\}', r'\\S\1', body)
    body = body.replace(' of the main paper--\\S', '--\\S')
    head = ('\\subsection*{%s}\n'
            '\\addcontentsline{toc}{subsection}{%s}\n'
            '\\noindent\\emph{Referred to from the main paper, %s.}\n\n'
            % (title, title, ptr))
    io.open('%s/%02d.tex' % (OUTDIR, i), 'w', encoding='utf-8').write(head + body + '\n')

print('wrote %d float shards to %s/' % (len(ORDER), OUTDIR))
print('tables: S1..S6 | figures: S1..S3')
