# -*- coding: utf-8 -*-
"""把 主论文_submission_v1.md 的 Appendix A/B/C 转成 LaTeX。
   要点：先按 $...$ 与 `...` 切出数学/代码跨度并原样保护，其余文本逐字符转义，
   避免 % & # _ 误伤以及 markdown 强调符号与 LaTeX 反斜杠互相污染。"""
import io, re

MD = '../../主论文_submission_v1.md'
OUT = 'appendix_src.tex'

PLAIN = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '#': '\\#',
         '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}',
         '^': '\\textasciicircum{}', '$': '\\$'}

GREEK = {'\u03b1': '\\alpha', '\u03b2': '\\beta', '\u03b3': '\\gamma',
         '\u03b4': '\\delta', '\u03b5': '\\varepsilon', '\u03b6': '\\zeta',
         '\u03b7': '\\eta', '\u03b8': '\\theta', '\u03ba': '\\kappa',
         '\u03bb': '\\lambda', '\u03bc': '\\mu', '\u00b5': '\\mu',
         '\u03bd': '\\nu', '\u03be': '\\xi', '\u03c0': '\\pi',
         '\u03c1': '\\rho', '\u03c3': '\\sigma', '\u03c4': '\\tau',
         '\u03c6': '\\varphi', '\u03c7': '\\chi', '\u03c8': '\\psi',
         '\u03c9': '\\omega', '\u0394': '\\Delta', '\u03a3': '\\Sigma',
         '\u03a6': '\\Phi', '\u03a9': '\\Omega', '\u039b': '\\Lambda'}

SYM = {'\u2014': '---', '\u2013': '--', '\u2026': '\\ldots{}',
       '\u00d7': '$\\times$', '\u2265': '$\\ge$', '\u2264': '$\\le$',
       '\u2248': '$\\approx$', '\u2260': '$\\ne$', '\u2192': '$\\to$',
       '\u2190': '$\\leftarrow$', '\u21d2': '$\\Rightarrow$',
       '\u2194': '$\\leftrightarrow$', '\u00b1': '$\\pm$',
       '\u2212': '$-$', '\u00b7': '\\textperiodcentered{}', '\u00a7': '\\S',
       '\u00b0': '$^\\circ$', '\u201c': '``', '\u201d': "''",
       '\u2018': '`', '\u2019': "'", '\u2713': '\\checkmark{}',
       '\u2032': "$'$", '\u2033': "''"}


DEDASH_TITLE = re.compile(r'(Appendix [A-Z]) --- ')
DEDASH_SENT = re.compile(r' --- ([A-Z])')
DEDASH_REST = re.compile(r' --- ')

# 降噪后少数会被逗号读乱的结构：把定义列表/解释从句改用括号、冒号或分号。
SPECIAL = [
    ("non-propositionality, questions, imperatives, opinion, speculation, accounts",
     "non-propositionality (questions, imperatives, opinion, speculation) accounts"),
    ("write the clause down explicitly, a leading mention",
     "write the clause down explicitly: a leading mention"),
    ("Combined with A2, content cannot recover the edge, this is a complete irreducibility",
     "Combined with A2 (content cannot recover the edge), this is a complete irreducibility"),
    ("with respect to the second, macro-F1 on the annotated set rises",
     "with respect to the second; macro-F1 on the annotated set rises"),
    ("mergeability was worth doing, it removes the last unclosed clause",
     "mergeability was worth doing: it removes the last unclosed clause"),
]


def dedash(s):
    """em-dash 降噪：把格式化的破折号换成学术写作常用标点。
       标题分隔 (Appendix X --- T) -> 冒号；后接独立分句 -> 分号；其余 -> 逗号。
       只改标点，不动任何词、数字与引用。"""
    s = DEDASH_TITLE.sub(r'\1: ', s)
    s = DEDASH_SENT.sub(r'; \1', s)
    s = DEDASH_REST.sub(', ', s)
    for a, b in SPECIAL:
        s = s.replace(a, b)
    return s


def mk(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == '`':
            j = s.find('`', i + 1)
            if j > 0:
                inner = s[i + 1:j]
                out.append('\\texttt{%s}' % ''.join(PLAIN.get(x, x) for x in inner))
                i = j + 1
                continue
        if c == '$':
            j = s.find('$', i + 1)
            if j > 0:
                out.append('$%s$' % s[i + 1:j].replace('%', '\\%'))
                i = j + 1
                continue
        if s.startswith('**', i):
            j = s.find('**', i + 2)
            if j > 0:
                out.append('\\textbf{%s}' % mk(s[i + 2:j]))
                i = j + 2
                continue
        if c == '*':
            j = s.find('*', i + 1)
            if j > 0:
                out.append('\\emph{%s}' % mk(s[i + 1:j]))
                i = j + 1
                continue
        if c in GREEK:
            out.append('$%s$' % GREEK[c])
            i += 1
            continue
        if c in SYM:
            out.append(SYM[c])
            i += 1
            continue
        out.append(PLAIN.get(c, c))
        i += 1
    return ''.join(out)


src = io.open(MD, encoding='utf-8').read().split('\n')

# 取第 579 行到 References 之前
try:
    a = next(i for i, l in enumerate(src) if l.startswith('## Appendix A'))
    b = next(i for i, l in enumerate(src) if l.startswith('## References'))
except StopIteration:
    raise SystemExit('未找到 Appendix A / References 边界')
body = src[a:b]
print('source lines %d..%d (%d lines)' % (a + 1, b, len(body)))

res, in_list = [], False
for raw in body:
    ln = raw.rstrip()
    t = ln.strip()
    if not t or t == '---':
        if in_list:
            res.append('\\end{itemize}')
            in_list = False
        if t == '':
            res.append('')
        continue
    if t.startswith('## '):
        if in_list:
            res.append('\\end{itemize}')
            in_list = False
        head = t[3:].strip()
        res.append('')
        res.append('\\section*{%s}' % mk(head))
        res.append('\\addcontentsline{toc}{section}{%s}' % mk(head))
        continue
    if t.startswith('### '):
        if in_list:
            res.append('\\end{itemize}')
            in_list = False
        res.append('\\subsection{%s}' % mk(t[4:].strip()))
        continue
    if t.startswith('- '):
        if not in_list:
            res.append('')
            res.append('\\begin{itemize}')
            in_list = True
        res.append('\\item %s' % mk(t[2:].strip()))
        continue
    if in_list:
        # 列表项续行
        res.append(mk(t))
        continue
    res.append(mk(t))

if in_list:
    res.append('\\end{itemize}')

tex = '\n'.join(res)
tex = re.sub(r'\n{3,}', '\n\n', tex)
tex = dedash(tex)
io.open(OUT, 'w', encoding='utf-8').write(tex)

# 体检：残留未转义危险字符
bad = []
for ch in ('&', '#', '%', '_'):
    for m in re.finditer(re.escape(ch), tex):
        ctx = tex[max(0, m.start() - 60):m.start() + 30].replace('\n', ' ')
        if '\\' + ch not in tex[max(0, m.start() - 1):m.start() + 1]:
            bad.append((ch, ctx))
print('written %s : %d bytes, %d lines' % (OUT, len(tex), tex.count('\n') + 1))
print('suspicious unescaped chars: %d' % len(bad))
for ch, ctx in bad[:12]:
    print('   [%s] ...%s...' % (ch, ctx))
