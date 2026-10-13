"""
Rendering helpers for chat messages.

`prepare_markdown` makes LLM output safe and displayable in `st.markdown`:
  1. Normalises LaTeX delimiters (\\( \\), \\[ \\], \\begin{equation}...) to the
     `$...$` / `$$...$$` forms that Streamlit's KaTeX renderer understands.
  2. Sanitises any embedded HTML with an allow-list (nh3) so that HTML the
     model returns (tables, <sup>, <br>, ...) renders without enabling script
     execution or other XSS vectors.
"""
from __future__ import annotations

import re

import nh3

# -- Allow-lists for HTML sanitising ----------------------------------------

_ALLOWED_TAGS = {
    "a", "b", "i", "u", "em", "strong", "sub", "sup", "br", "hr", "p", "span",
    "div", "blockquote", "pre", "code", "ul", "ol", "li", "h1", "h2", "h3",
    "h4", "h5", "h6", "table", "thead", "tbody", "tfoot", "tr", "th", "td",
    "caption", "details", "summary", "mark", "small", "del", "ins",
}
_ALLOWED_ATTRS = {
    "a": {"href", "title"},
    "th": {"colspan", "rowspan", "align"},
    "td": {"colspan", "rowspan", "align"},
}
_ALLOWED_SCHEMES = {"http", "https", "mailto"}

# -- Regexes ----------------------------------------------------------------

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_DISPLAY_MATH = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\\)\$(?!\s)[^$\n]+?(?<!\s)\$")
_KNOWN_TAGS = "|".join(sorted(_ALLOWED_TAGS | {"script", "style", "iframe", "img", "svg", "object", "embed", "form", "input", "button"}))
_ATTR = r"""(?:\s+[\w:-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))*"""
_ALLOWED_TAG_START = re.compile(
    r"<(?!/?(?:" + "|".join(sorted(_ALLOWED_TAGS)) + r")" + _ATTR + r"\s*/?>)"
)
_HAS_HTML = re.compile(rf"</?(?:{_KNOWN_TAGS})(?:\s[^<>]*)?/?>", re.IGNORECASE)

_BRACKET_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_PAREN_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_ENV_DISPLAY = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.+?)\\end\{\1\}",
    re.DOTALL,
)


def normalize_latex(text: str) -> str:
    """Convert alternative LaTeX delimiters to $...$ / $$...$$ (code is left untouched)."""
    stash: list[str] = []

    def _hide(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"CODEPLACEHOLDER{len(stash) - 1}END"

    text = _FENCED_CODE.sub(_hide, text)
    text = _INLINE_CODE.sub(_hide, text)

    text = _BRACKET_DISPLAY.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    text = _PAREN_INLINE.sub(lambda m: f"${m.group(1).strip()}$", text)

    def _env(m: re.Match) -> str:
        name, body = m.group(1).rstrip("*"), m.group(2).strip()
        if name in ("equation", "multline"):
            return f"\n$$\n{body}\n$$\n"
        # KaTeX needs the aligned/gathered variants inside $$...$$
        inner = "aligned" if name == "align" else "gathered"
        return f"\n$$\n\\begin{{{inner}}}{body}\\end{{{inner}}}\n$$\n"

    text = _ENV_DISPLAY.sub(_env, text)

    for i, original in enumerate(stash):
        text = text.replace(f"CODEPLACEHOLDER{i}END", original)
    return text


def sanitize_html(text: str) -> str:
    """Sanitise HTML embedded in markdown while leaving math and code spans untouched."""
    if not _HAS_HTML.search(text):
        return text

    stash: list[str] = []

    def _hide(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"KEEPPLACEHOLDER{len(stash) - 1}END"

    for pattern in (_FENCED_CODE, _INLINE_CODE, _DISPLAY_MATH, _INLINE_MATH):
        text = pattern.sub(_hide, text)

    # A '<' that does not open/close an allowed tag is literal text (e.g. "x<y"):
    # escape it so the sanitiser does not swallow the surrounding content.
    text = _ALLOWED_TAG_START.sub("&lt;", text)

    text = nh3.clean(
        text,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        url_schemes=_ALLOWED_SCHEMES,
        link_rel="noopener noreferrer",
    )

    # Restore in reverse so nested placeholders (none expected) resolve correctly
    for i in range(len(stash) - 1, -1, -1):
        text = text.replace(f"KEEPPLACEHOLDER{i}END", stash[i])
    return text


def prepare_markdown(text: str) -> str:
    """Full pipeline: LaTeX delimiter normalisation, then HTML sanitising."""
    return sanitize_html(normalize_latex(text or ""))
