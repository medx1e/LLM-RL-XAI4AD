"""
Reusable HTML component builders for the XAI4AD Streamlit UI.

All functions return HTML strings ready for:
    st.markdown(html, unsafe_allow_html=True)

Uses CSS utility classes injected by theme.py (panel, chip, glass, grid-bg,
scanline, mono, xai-*). No Streamlit imports — pure string builders.
"""

from __future__ import annotations

import html as _html


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ─── Sidebar brand ────────────────────────────────────────────────────────────

def sidebar_brand(version: str = "v0.4", branch: str = "llm") -> str:
    git_icon = (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/>'
        '<circle cx="6" cy="18" r="3"/>'
        '<path d="M18 9a9 9 0 0 1-9 9"/><line x1="6" y1="9" x2="6" y2="15"/>'
        '</svg>'
    )
    return (
        f"<div style='padding:16px 12px 12px;border-bottom:1px solid var(--sidebar-border,#2c2f45);"
        f"margin-bottom:8px;'>"
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"<div style='width:32px;height:32px;border-radius:8px;"
        f"background:rgba(111,106,232,0.15);color:#6f6ae8;"
        f"display:flex;align-items:center;justify-content:center;flex-shrink:0;'>"
        f"{git_icon}</div>"
        f"<div>"
        f"<div style='font-size:14px;font-weight:700;color:#e8e8ef;line-height:1.2;'>XAI4AD</div>"
        f"<div class='mono' style='font-size:10px;color:#a4a5b0;margin-top:1px;'>"
        f"{version} · {branch}</div>"
        f"</div></div></div>"
    )


# ─── Context strip (sticky page header) ──────────────────────────────────────

def context_strip(
    tab_name: str,
    model_label: str | None = None,
    scenario_label: str | None = None,
) -> str:
    crumbs = []
    if model_label:
        crumbs.append(f"<span style='font-size:13px;color:var(--muted-fg,#a4a5b0);'>{model_label}</span>")
    if scenario_label:
        crumbs.append(f"<span style='font-size:13px;color:var(--muted-fg,#a4a5b0);'>{scenario_label}</span>")
    sep = "<span style='color:var(--border,#393d57);margin:0 6px;'>›</span>"
    breadcrumb = sep.join(crumbs)
    trail = (
        f"<span style='color:var(--border,#393d57);margin:0 10px;'>·</span>{breadcrumb}"
        if breadcrumb else ""
    )
    return (
        f"<div class='xai-context-strip'>"
        f"<h2 style='margin:0;font-size:17px;font-weight:700;"
        f"color:var(--foreground,#f5f5f8);letter-spacing:-0.01em;"
        f"line-height:1;font-family:Inter,sans-serif;'>{tab_name}</h2>"
        f"{trail}"
        f"</div>"
    )


# ─── Hero block (Home tab) ────────────────────────────────────────────────────

def hero(
    eyebrow: str,
    title: str,
    subtitle: str,
    version: str = "v0.4",
    branch: str = "llm",
    n_scenarios: int | None = None,
) -> str:
    chip_preview = chip("Research preview", color="primary")
    chip_ver = chip(f"{version} · {branch}", mono=True)
    chip_scen = ""
    if n_scenarios:
        noun = "scenario" if n_scenarios == 1 else "scenarios"
        chip_scen = chip(f"{n_scenarios} curated {noun}", mono=True)
    eyebrow_html = (
        f"<div class='xai-eyebrow' style='margin-bottom:16px;font-size:12px;'>{eyebrow}</div>"
        if eyebrow else ""
    )
    return (
        f"<div class='xai-slide-up xai-hero' "
        f"style='text-align:center;padding:56px 24px 48px;margin-bottom:8px;'>"
        # ── eyebrow ──
        f"{eyebrow_html}"
        # ── title ──
        f"<h1 style='margin:0 0 22px;font-size:46px;font-weight:800;"
        f"color:var(--foreground,#f5f5f8);letter-spacing:-0.03em;"
        f"line-height:1.12;font-family:Inter,sans-serif;'>{title}</h1>"
        # ── subtitle (editorial serif) ──
        f"<p style='margin:0 auto;font-size:19px;line-height:1.65;"
        f"color:var(--muted-fg,#a4a5b0);font-family:\"Source Serif 4\",Georgia,serif;"
        f"max-width:660px;'>{subtitle}</p>"
        # ── chip row ──
        f"<div style='display:inline-flex;flex-wrap:wrap;align-items:center;"
        f"justify-content:center;gap:10px;margin-top:28px;'>"
        f"{chip_preview}{chip_ver}{chip_scen}"
        f"</div>"
        # ── gradient accent line ──
        f"<div style='margin:32px auto 0;width:120px;height:3px;"
        f"border-radius:999px;"
        f"background:linear-gradient(90deg,transparent,var(--primary,#6f6ae8),transparent);"
        f"opacity:0.55;'></div>"
        f"</div>"
    )


# ─── Section badge (lettered) ─────────────────────────────────────────────────

def section_badge(
    letter: str,
    title: str,
    hint: str | None = None,
    large: bool = False,
) -> str:
    hint_sz     = "13px" if large else "12px"
    title_sz    = "17px" if large else "15px"
    margin      = "32px 0 12px" if large else "20px 0 8px"
    badge_style = (
        "width:26px;height:26px;font-size:13px;" if large else ""
    )
    hint_html = (
        f"<span style='font-size:{hint_sz};color:var(--muted-fg,#a4a5b0);"
        f"font-weight:400;margin-left:8px;'>{hint}</span>"
        if hint else ""
    )
    return (
        f"<div style='display:flex;align-items:center;gap:10px;margin:{margin};'>"
        f"<span class='xai-section-badge' style='{badge_style}'>{letter}</span>"
        f"<h2 style='margin:0;font-size:{title_sz};font-weight:600;"
        f"color:var(--foreground,#f5f5f8);letter-spacing:-0.005em;"
        f"font-family:Inter,sans-serif;'>{title}{hint_html}</h2>"
        f"</div>"
    )


# ─── Section title (compact) ─────────────────────────────────────────────────

def section_title(label: str, hint: str | None = None) -> str:
    h = (
        f"<span style='font-size:12px;color:var(--muted-fg,#a4a5b0);"
        f"margin-left:10px;font-weight:400;'>{hint}</span>"
        if hint else ""
    )
    return (
        f"<h3 style='margin:18px 0 12px;font-size:15px;font-weight:700;"
        f"color:var(--foreground,#f5f5f8);font-family:Inter,sans-serif;"
        f"letter-spacing:-0.01em;'>{label}{h}</h3>"
    )


# ─── Section description ──────────────────────────────────────────────────────

def section_desc(text: str) -> str:
    return (
        f"<p style='font-size:13px;color:var(--muted-fg,#a4a5b0);"
        f"margin:-4px 0 14px;line-height:1.55;font-family:Inter,sans-serif;'>{text}</p>"
    )


# ─── Chip / badge ─────────────────────────────────────────────────────────────

def chip(label: str, mono: bool = False, color: str | None = None) -> str:
    cls = "chip"
    if mono:
        cls += " chip-mono"
    if color == "primary":
        cls += " chip-primary"
    elif color == "success":
        cls += " chip-success"
    elif color == "warning":
        cls += " chip-warning"
    return f"<span class='{cls}'>{label}</span>"


_TONE_COLORS: dict[str, str] = {
    "detailed": "#DC2626", "brief": "#10B981",
    "caveat": "#F59E0B", "detailed_caveat": "#7C3AED", "detailed-caveat": "#7C3AED",
}
_TONE_LABELS: dict[str, str] = {
    "detailed": "Detailed", "brief": "Brief",
    "caveat": "Caveat", "detailed_caveat": "Detailed + Caveat", "detailed-caveat": "Caveat+",
}


def tone_badge(tone: str) -> str:
    color = _TONE_COLORS.get(tone, "#6f6ae8")
    label = _TONE_LABELS.get(tone, tone.replace("_", " ").title())
    bg     = _rgba(color, 0.16)
    border = _rgba(color, 0.38)
    return (
        f"<span style='display:inline-flex;align-items:center;gap:5px;"
        f"padding:2px 10px;border-radius:999px;"
        f"font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;"
        f"color:{color};background:{bg};border:1px solid {border};'>"
        f"<span style='width:5px;height:5px;border-radius:50%;background:{color};"
        f"flex-shrink:0;display:inline-block;'></span>"
        f"{label}</span>"
    )


# ─── Panel wrapper ────────────────────────────────────────────────────────────

def panel(inner_html: str, glass: bool = False, padding: str = "20px 24px") -> str:
    cls = "glass" if glass else "panel"
    return f"<div class='{cls}' style='padding:{padding};'>{inner_html}</div>"


# ─── Metric tile ─────────────────────────────────────────────────────────────

_VALUE_COLORS = {
    "success": "var(--success,#2fbe87)",
    "warning": "var(--warning,#f5c045)",
    "error":   "var(--error,#e05249)",
    "muted":   "var(--muted-fg,#a4a5b0)",
    "primary": "var(--primary,#6f6ae8)",
}


def metric_card(
    label: str,
    value: str,
    hint: str | None = None,
    color_key: str | None = None,
    icon: str | None = None,
    top_accent: str | None = None,
    large: bool = False,
) -> str:
    value_color = _VALUE_COLORS.get(color_key or "", "var(--foreground,#f5f5f8)")
    border_top = f"border-top:2px solid {top_accent};" if top_accent else ""
    pad        = "22px" if large else "16px"
    label_sz   = "13px" if large else "11px"
    value_sz   = "30px" if large else "22px"
    hint_sz    = "12px" if large else "11px"
    icon_html = (
        f"<div style='font-size:16px;margin-bottom:4px;color:var(--muted-fg,#a4a5b0);'>{icon}</div>"
        if icon else ""
    )
    hint_html = (
        f"<div style='margin-top:6px;font-size:{hint_sz};color:var(--muted-fg,#a4a5b0);'>{hint}</div>"
        if hint else ""
    )
    return (
        f"<div class='panel xai-fade-in' style='padding:{pad};{border_top}'>"
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;'>"
        f"<span style='font-size:{label_sz};font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.12em;color:var(--muted-fg,#a4a5b0);"
        f"font-family:Inter,sans-serif;'>{label}</span>"
        f"{icon_html}"
        f"</div>"
        f"<div style='margin-top:10px;font-size:{value_sz};font-weight:700;color:{value_color};"
        f"font-variant-numeric:tabular-nums;font-family:Inter,sans-serif;'>{value}</div>"
        f"{hint_html}"
        f"</div>"
    )


def metric_row(cards: list[tuple[str, str, str | None]]) -> str:
    cols = "".join(
        f"<div style='flex:1;min-width:0;'>{metric_card(label, value, hint)}</div>"
        for label, value, hint in cards
    )
    return f"<div style='display:flex;gap:12px;margin-bottom:16px;'>{cols}</div>"


# ─── Feature card (Home tab) ──────────────────────────────────────────────────

def feature_card(
    color: str,
    title: str,
    desc: str,
    icon_char: str = "◬",
    tag: str | None = None,       # retained for compat; navigation now uses a real button
    min_height: str = "184px",    # fixed card height so all cards align regardless of copy
) -> str:
    icon_bg = _rgba(color, 0.16) if color.startswith("#") else "rgba(111,106,232,0.16)"
    return (
        f"<div class='panel panel-interactive xai-fade-in' "
        f"style='padding:22px;border-left:3px solid {color};"
        f"min-height:{min_height};box-sizing:border-box;'>"
        f"<div style='display:flex;align-items:flex-start;gap:16px;'>"
        f"<div style='width:46px;height:46px;border-radius:12px;flex-shrink:0;"
        f"font-size:22px;background:{icon_bg};color:{color};"
        f"display:flex;align-items:center;justify-content:center;'>"
        f"{icon_char}</div>"
        f"<div style='flex:1;min-width:0;'>"
        f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
        f"<h3 style='margin:0;font-size:16px;font-weight:600;"
        f"color:var(--foreground,#f5f5f8);font-family:Inter,sans-serif;'>{title}</h3>"
        f"<span class='feature-arrow' style='color:{color};font-size:15px;'>→</span>"
        f"</div>"
        f"<p style='margin:7px 0 0;font-size:13.5px;line-height:1.6;"
        f"color:var(--muted-fg,#a4a5b0);font-family:Inter,sans-serif;'>{desc}</p>"
        f"</div></div></div>"
    )


# ─── Scenario card (Home tab) ─────────────────────────────────────────────────

def scenario_card(
    scenario_id: str,
    model_name: str,
    encoder_family: str,
    has_attention: bool,
    description: str,
    delay_class: str = "",
) -> str:
    attn_chip = chip("attention", color="primary") if has_attention else ""
    enc_chip = chip(encoder_family, mono=True)
    desc_short = description[:88].rstrip() + "…" if len(description) > 90 else description
    return (
        f"<div class='panel panel-interactive xai-fade-in {delay_class}' "
        f"style='padding:18px;height:100%;'>"
        f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:14px;'>"
        f"{enc_chip}{attn_chip}</div>"
        f"<div class='mono' style='font-size:12px;color:var(--muted-fg,#a4a5b0);"
        f"margin-bottom:5px;'>Scenario {scenario_id}</div>"
        f"<div style='font-size:15px;font-weight:600;"
        f"color:var(--foreground,#f5f5f8);margin-bottom:9px;line-height:1.3;'>{model_name}</div>"
        f"<div style='font-size:12.5px;color:var(--muted-fg,#a4a5b0);"
        f"line-height:1.6;min-height:40px;'>{desc_short}</div>"
        f"</div>"
    )


# ─── Narration card ───────────────────────────────────────────────────────────

def narration_card(
    model_name: str,
    subtitle: str,
    tone: str,
    text: str,
    step: int,
    response_time_ms: float | None = None,
    decision: str | None = None,
) -> str:
    tone_color = _TONE_COLORS.get(tone, "#6f6ae8")
    rt_html = (
        f"<span class='mono'>{response_time_ms:.0f} ms</span>"
        f"<span style='color:var(--border,#393d57);'>·</span>"
        if response_time_ms else ""
    )
    dec_html = (
        f"<span>decision: <b style='color:rgba(245,245,248,.85);'>{decision}</b></span>"
        if decision else ""
    )
    return (
        f"<article class='panel xai-fade-in' "
        f"style='overflow:hidden;border-left:3px solid {tone_color};margin-bottom:12px;'>"
        f"<header style='display:flex;align-items:center;gap:10px;"
        f"padding:12px 16px;border-bottom:1px solid var(--border,#393d57);'>"
        f"<div style='min-width:0;'>"
        f"<div style='font-size:14px;font-weight:600;line-height:1.2;"
        f"color:var(--foreground,#f5f5f8);'>{model_name}</div>"
        f"<div class='mono' style='font-size:11px;color:var(--muted-fg,#a4a5b0);"
        f"margin-top:2px;'>{subtitle}</div>"
        f"</div>"
        f"<div style='margin-left:auto;flex-shrink:0;'>{tone_badge(tone)}</div>"
        f"</header>"
        f"<div style='padding:16px;'>"
        f"<p style='font-family:\"Source Serif 4\",Georgia,serif;"
        f"font-size:15px;line-height:1.65;color:rgba(245,245,248,.95);margin:0;'>{text}</p>"
        f"<div style='margin-top:12px;display:flex;flex-wrap:wrap;align-items:center;gap:8px;"
        f"font-size:11px;color:var(--muted-fg,#a4a5b0);'>"
        f"<span class='mono'>t = {step}</span>"
        f"<span style='color:var(--border,#393d57);'>·</span>"
        f"{rt_html}{dec_html}"
        f"</div></div></article>"
    )


def narration_card_streaming(
    model_name: str,
    subtitle: str,
    tone: str,
    text: str,
    step: int,
    response_time_ms: float | None = None,
    decision: str | None = None,
    variant: int = 0,
    word_stagger_s: float = 0.045,
    fade_dur_s: float = 0.30,
    max_total_s: float = 3.5,
) -> str:
    """Same visual as ``narration_card`` but reveals the body token-by-token.

    Each word is wrapped in a span with a staggered ``animation-delay`` so the
    paragraph "types" itself in (LLM-streaming flourish for live demos).

    ``variant`` must flip (0 ↔ 1) every time a *new* narration is shown and stay
    constant while one narration is forward-filled. It picks between two
    identical keyframes (``token-in`` / ``token-in-2``); swapping the
    animation-name is what restarts the reveal, because Streamlit's react-markdown
    patches the span nodes in place across reruns rather than recreating them.
    Holding ``variant`` constant within a narration keeps the HTML byte-identical,
    so the animation runs once then freezes — no flicker during BEV playback.
    """
    tone_color = _TONE_COLORS.get(tone, "#6f6ae8")
    anim_name = "token-in" if variant % 2 == 0 else "token-in-2"
    wrapper_tag = "article" if variant % 2 == 0 else "section"
    rt_html = (
        f"<span class='mono'>{response_time_ms:.0f} ms</span>"
        f"<span style='color:var(--border,#393d57);'>·</span>"
        if response_time_ms else ""
    )
    dec_html = (
        f"<span>decision: <b style='color:rgba(245,245,248,.85);'>{decision}</b></span>"
        if decision else ""
    )

    words = text.split()
    n = max(len(words), 1)
    # Compress the stagger for long narrations so the reveal never drags on.
    stagger = min(word_stagger_s, max_total_s / n)
    spans = "".join(
        f"<span style='animation:{anim_name} {fade_dur_s}s var(--ease-out,ease) both;"
        f"animation-delay:{i * stagger:.3f}s;'>{_html.escape(w)} </span>"
        for i, w in enumerate(words)
    )
    caret = (
        f"<span style='display:inline-block;width:2px;height:1.05em;"
        f"vertical-align:-0.16em;margin-left:1px;background:{tone_color};"
        f"animation:caret-blink 0.9s steps(1) infinite;"
        f"animation-delay:{n * stagger:.3f}s;opacity:0;'></span>"
    )

    return (
        f"<{wrapper_tag} class='panel xai-fade-in' "
        f"style='overflow:hidden;border-left:3px solid {tone_color};margin-bottom:12px;'>"
        f"<header style='display:flex;align-items:center;gap:10px;"
        f"padding:12px 16px;border-bottom:1px solid var(--border,#393d57);'>"
        f"<div style='min-width:0;'>"
        f"<div style='font-size:14px;font-weight:600;line-height:1.2;"
        f"color:var(--foreground,#f5f5f8);'>{model_name}</div>"
        f"<div class='mono' style='font-size:11px;color:var(--muted-fg,#a4a5b0);"
        f"margin-top:2px;'>{subtitle}</div>"
        f"</div>"
        f"<div style='margin-left:auto;flex-shrink:0;'>{tone_badge(tone)}</div>"
        f"</header>"
        f"<div style='padding:16px;'>"
        f"<p style='font-family:\"Source Serif 4\",Georgia,serif;"
        f"font-size:15px;line-height:1.65;color:rgba(245,245,248,.95);margin:0;'>"
        f"{spans}{caret}</p>"
        f"<div style='margin-top:12px;display:flex;flex-wrap:wrap;align-items:center;gap:8px;"
        f"font-size:11px;color:var(--muted-fg,#a4a5b0);'>"
        f"<span class='mono'>t = {step}</span>"
        f"<span style='color:var(--border,#393d57);'>·</span>"
        f"{rt_html}{dec_html}"
        f"</div></div></{wrapper_tag}>"
    )


# ─── Method accent bar ────────────────────────────────────────────────────────

_METHOD_COLORS_LIST: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def method_accent_bar(method_name: str, method_idx: int = 0) -> str:
    color = _METHOD_COLORS_LIST[method_idx % len(_METHOD_COLORS_LIST)]
    return (
        f"<div style='height:3px;background:{color};"
        f"border-radius:2px 2px 0 0;margin-bottom:-1px;opacity:0.9;'></div>"
    )


# ─── HTML heatmap table ───────────────────────────────────────────────────────

def heatmap_table(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    diverging: bool = True,
    fmt: str = ".2f",
    row_colors: list[str] | None = None,
) -> str:
    def cell_bg(v: float) -> str:
        if v != v:
            return "rgba(57,61,87,0.3)"
        clamp = max(-1.0, min(1.0, v))
        if not diverging:
            pct = max(0.05, min(0.85, abs(v)))
            return f"rgba(91,108,249,{pct:.2f})"
        if clamp >= 0:
            return f"rgba(239,68,68,{clamp*0.72:.2f})"
        return f"rgba(91,108,249,{-clamp*0.72:.2f})"

    def cell_text(v: float) -> str:
        return f"{v:{fmt}}" if v == v else "—"

    th_style = (
        "padding:4px 8px;font-size:11px;font-weight:500;"
        "color:var(--muted-fg,#a4a5b0);font-family:'JetBrains Mono',monospace;"
    )
    col_headers = "".join(
        f"<th style='{th_style}text-align:center;min-width:60px;'>{c}</th>"
        for c in col_labels
    )
    rows_html = ""
    for ri, (row_label, row) in enumerate(zip(row_labels, matrix)):
        label_color = row_colors[ri] if row_colors and ri < len(row_colors) else "var(--muted-fg,#a4a5b0)"
        cells = "".join(
            f"<td title='ρ = {cell_text(v)}'"
            f" style='min-width:60px;height:40px;text-align:center;"
            f"font-size:11px;font-weight:500;font-variant-numeric:tabular-nums;"
            f"border-radius:4px;background:{cell_bg(v)};"
            f"outline:1px solid var(--border,#393d57);"
            f"color:var(--foreground,#f5f5f8);cursor:default;'>"
            f"{cell_text(v)}</td>"
            for v in row
        )
        rows_html += (
            f"<tr><th style='{th_style}text-align:right;color:{label_color};"
            f"padding-right:10px;white-space:nowrap;'>{row_label}</th>{cells}</tr>"
        )

    title_html = (
        f"<div style='font-size:12px;font-weight:600;color:var(--foreground,#f5f5f8);"
        f"margin-bottom:10px;font-family:Inter,sans-serif;text-transform:uppercase;"
        f"letter-spacing:0.06em;'>{title}</div>"
        if title else ""
    )
    return (
        f"<div class='panel' style='padding:16px;'>"
        f"{title_html}"
        f"<div style='overflow-x:auto;'>"
        f"<table style='border-collapse:separate;border-spacing:2px;'>"
        f"<thead><tr><th style='{th_style}'></th>{col_headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div></div>"
    )


# ─── Warning / info banners ───────────────────────────────────────────────────

def warning_banner(message: str, code: str | None = None) -> str:
    warn_icon = (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="flex-shrink:0;margin-top:1px;">'
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
        '</svg>'
    )
    code_html = (
        f"<code style='display:block;margin-top:8px;font-family:\"JetBrains Mono\",monospace;"
        f"font-size:11px;color:var(--foreground,#f5f5f8);background:var(--secondary,#2e3147);"
        f"padding:6px 10px;border-radius:6px;border:1px solid var(--border,#393d57);'>{code}</code>"
        if code else ""
    )
    return f"<div class='xai-warn'>{warn_icon}<span>{message}{code_html}</span></div>"


def info_banner(message: str, code: str | None = None) -> str:
    info_icon = (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="flex-shrink:0;margin-top:1px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
        '</svg>'
    )
    code_html = (
        f"<code style='display:block;margin-top:8px;font-family:\"JetBrains Mono\",monospace;"
        f"font-size:11px;color:var(--foreground,#f5f5f8);background:var(--secondary,#2e3147);"
        f"padding:6px 10px;border-radius:6px;border:1px solid var(--border,#393d57);'>{code}</code>"
        if code else ""
    )
    return f"<div class='xai-info'>{info_icon}<span>{message}{code_html}</span></div>"


# ─── Method profile card (Evaluation) ─────────────────────────────────────────

def method_profile_card(
    name: str,
    color: str,
    sparsity: str,
    gini: str,
    topk: str,
    stability: str,
    dominant: str,
) -> str:
    def _row(label: str, val: str) -> str:
        return (
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:baseline;padding:5px 0;"
            f"border-bottom:1px solid rgba(57,61,87,0.5);'>"
            f"<span style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;"
            f"color:var(--muted-fg,#a4a5b0);font-weight:600;'>{label}</span>"
            f"<span style='font-size:14px;font-weight:700;font-variant-numeric:tabular-nums;"
            f"color:var(--foreground,#f5f5f8);'>{val}</span>"
            f"</div>"
        )
    return (
        f"<div class='panel xai-fade-in' style='padding:14px;border-top:2px solid {color};'>"
        f"<div style='font-size:12px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:{color};margin-bottom:12px;"
        f"font-family:Inter,sans-serif;'>{name.replace('_', ' ')}</div>"
        f"{_row('Sparsity', sparsity)}"
        f"{_row('Gini', gini)}"
        f"{_row('Top-10%', topk)}"
        f"{_row('Stability', stability)}"
        f"<div style='margin-top:8px;font-size:11px;color:var(--muted-fg,#a4a5b0);'>"
        f"Dominates: <span style='color:var(--foreground,#f5f5f8);'>{dominant}</span></div>"
        f"</div>"
    )


# ─── Empty state ──────────────────────────────────────────────────────────────

_SVG_EMPTY = (
    '<svg width="36" height="36" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<line x1="9" y1="9" x2="15" y2="9"/>'
    '<line x1="9" y1="12" x2="15" y2="12"/>'
    '<line x1="9" y1="15" x2="13" y2="15"/>'
    '</svg>'
)


def empty_state(
    message: str,
    icon: str | None = None,
    hint: str | None = None,
    command: str | None = None,
) -> str:
    icon_html = icon if icon else _SVG_EMPTY
    hint_html = (
        f"<p style='font-size:12px;color:var(--muted-fg,#a4a5b0);margin:6px 0 0;'>{hint}</p>"
        if hint else ""
    )
    cmd_html = (
        f"<code style='display:block;margin-top:10px;"
        f"font-family:\"JetBrains Mono\",monospace;font-size:11px;"
        f"color:var(--foreground,#f5f5f8);background:var(--secondary,#2e3147);"
        f"padding:6px 10px;border-radius:6px;border:1px solid var(--border,#393d57);"
        f"text-align:left;'>{command}</code>"
        if command else ""
    )
    return (
        f"<div style='text-align:center;padding:40px 24px;"
        f"background:rgba(111,106,232,0.04);"
        f"border:1px dashed rgba(111,106,232,0.22);"
        f"border-radius:var(--radius,12px);color:var(--muted-fg,#a4a5b0);'>"
        f"<div style='display:flex;justify-content:center;margin-bottom:12px;"
        f"opacity:0.55;'>{icon_html}</div>"
        f"<p style='font-size:14px;font-weight:500;color:var(--foreground,#f5f5f8);"
        f"margin:0;'>{message}</p>"
        f"{hint_html}{cmd_html}"
        f"</div>"
    )


# ─── Skeleton loader ──────────────────────────────────────────────────────────

def skeleton(width: str = "100%", height: str = "200px") -> str:
    return (
        f"<div class='xai-skeleton' "
        f"style='width:{width};height:{height};border-radius:8px;'></div>"
    )


# ─── Control rail wrapper ─────────────────────────────────────────────────────

def control_rail_open(title: str = "Controls") -> str:
    return (
        f"<div class='panel xai-fade-in' style='padding:14px;margin-top:8px;'>"
        f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.14em;color:var(--muted-fg,#a4a5b0);margin-bottom:12px;"
        f"font-family:Inter,sans-serif;'>{title}</div>"
    )


def control_rail_close() -> str:
    return "</div>"


# ─── Section header with description ─────────────────────────────────────────

def section_header(title: str, subtitle: str | None = None) -> str:
    sub = (
        f"<p style='margin:4px 0 0;font-size:13px;color:var(--muted-fg,#a4a5b0);"
        f"line-height:1.5;'>{subtitle}</p>"
        if subtitle else ""
    )
    return (
        f"<div style='margin-bottom:16px;' class='xai-fade-in'>"
        f"<h3 style='margin:0;font-size:17px;font-weight:700;"
        f"color:var(--foreground,#f5f5f8);letter-spacing:-0.01em;"
        f"font-family:Inter,sans-serif;'>{title}</h3>"
        f"{sub}</div>"
    )
