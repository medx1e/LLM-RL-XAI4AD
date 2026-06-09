"""
XAI4AD design system — ported from the React/Tailwind template in src/.

Call ``inject_theme()`` once in app.py immediately after ``st.set_page_config``.
"""

from __future__ import annotations

import streamlit as st

# ─── Google Fonts ─────────────────────────────────────────────────────────────
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400'
    '&family=JetBrains+Mono:wght@400;500;700'
    '&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap" rel="stylesheet">'
)

# ─── CSS design tokens (:root) ────────────────────────────────────────────────
TOKENS_CSS = """
:root {
  --font-sans:  "Inter", system-ui, -apple-system, sans-serif;
  --font-mono:  "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --font-serif: "Source Serif 4", Georgia, serif;

  --radius:      0.75rem;
  --radius-sm:   0.5rem;
  --radius-md:   calc(0.75rem - 2px);
  --radius-lg:   0.75rem;
  --radius-xl:   calc(0.75rem + 4px);
  --radius-pill: 999px;

  --background:  #1a1c28;
  --foreground:  #f5f5f8;
  --card:        #23263a;
  --popover:     #23263a;
  --secondary:   #2e3147;
  --muted:       #2a2d42;
  --muted-fg:    #a4a5b0;
  --border:      #393d57;
  --input:       #2e3147;

  --primary:     #6f6ae8;
  --primary-fg:  #ffffff;
  --accent:      #6f6ae8;

  --success:     #2fbe87;
  --warning:     #f5c045;
  --error:       #e05249;

  /* XAI semantic domain palette */
  --sdc:            #4C72B0;
  --agents-c:       #DD8452;
  --roadgraph:      #55A868;
  --traffic-lights: #C44E52;
  --gps:            #8172B2;

  /* 8 agent identity slots */
  --a0: #E41A1C; --a1: #377EB8; --a2: #4DAF4A; --a3: #984EA3;
  --a4: #FF7F00; --a5: #FFD92F; --a6: #A65628; --a7: #F781BF;

  /* Narration tones */
  --tone-detailed: #DC2626;
  --tone-brief:    #10B981;
  --tone-caveat:   #F59E0B;
  --tone-dc:       #7C3AED;

  /* Sidebar */
  --sidebar:        #1c1e2c;
  --sidebar-fg:     #e8e8ef;
  --sidebar-border: #2c2f45;
  --sidebar-accent: #282b3e;
  --sidebar-primary: #6f6ae8;

  --shadow-panel:
    0 1px 0 rgba(255,255,255,0.025) inset,
    0 8px 24px rgba(0,0,0,0.28);
  --shadow-pop: 0 12px 32px rgba(0,0,0,0.5);

  --ease-out: cubic-bezier(0.16,1,0.3,1);
  --dur-fast: 120ms;
  --dur-base: 180ms;
  --dur-slow: 300ms;
}
"""

# ─── Global reset ─────────────────────────────────────────────────────────────
GLOBAL_CSS = """
*, *::before, *::after { box-sizing: border-box; border-color: var(--border); }

::-webkit-scrollbar              { width: 10px; height: 10px; }
::-webkit-scrollbar-track        { background: transparent; }
::-webkit-scrollbar-thumb        { background: var(--border); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover  { background: var(--muted-fg); }

html, body {
  font-family: var(--font-sans);
  background: var(--background);
  color: var(--foreground);
  font-feature-settings: "cv02","cv03","cv04","ss01";
}
code, kbd { font-family: var(--font-mono); }
"""

# ─── Design utility classes (mirrors React template) ─────────────────────────
UTILITY_CSS = """
/* ── Panel / card ── */
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow-panel);
}

/* ── Glass ── */
.glass {
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: var(--radius);
}

/* ── Chip / badge ── */
.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--secondary);
  color: var(--foreground);
  vertical-align: middle;
  line-height: 1.6;
  white-space: nowrap;
}
.chip-mono {
  font-family: var(--font-mono) !important;
  text-transform: none !important;
  letter-spacing: normal !important;
}
.chip-primary {
  background: rgba(111,106,232,0.18) !important;
  color: #a8a3ff !important;
  border-color: rgba(111,106,232,0.4) !important;
}
.chip-success {
  background: rgba(47,190,135,0.14) !important;
  color: var(--success) !important;
  border-color: rgba(47,190,135,0.35) !important;
}
.chip-warning {
  background: rgba(245,192,69,0.14) !important;
  color: var(--warning) !important;
  border-color: rgba(245,192,69,0.35) !important;
}

/* ── Grid background ── */
.grid-bg {
  background-image:
    linear-gradient(to right, rgba(255,255,255,0.04) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(255,255,255,0.04) 1px, transparent 1px);
  background-size: 32px 32px;
}

/* ── Scanline overlay ── */
.scanline { position: relative; }
.scanline::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: repeating-linear-gradient(
    to bottom,
    transparent 0, transparent 3px,
    rgba(255,255,255,0.015) 3px, rgba(255,255,255,0.015) 4px
  );
  border-radius: inherit;
}

/* ── Mono helper ── */
.mono { font-family: var(--font-mono) !important; }

/* ── Colour helpers ── */
.text-primary  { color: var(--primary)  !important; }
.text-success  { color: var(--success)  !important; }
.text-warning  { color: var(--warning)  !important; }
.text-error    { color: var(--error)    !important; }
.text-muted    { color: var(--muted-fg) !important; }

/* ── Code snippet ── */
.xai-code {
  font-family: var(--font-mono);
  font-size: 11px;
  background: var(--secondary);
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--border);
  color: var(--foreground);
  display: inline-block;
  line-height: 1.6;
}

/* ── Section badge (A, B, C...) ── */
.xai-section-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  background: rgba(111,106,232,0.15);
  color: var(--primary);
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 700;
  flex-shrink: 0;
}

/* ── Warning / info banners ── */
.xai-warn {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(245,192,69,0.07);
  border: 1px solid rgba(245,192,69,0.28);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--warning);
  line-height: 1.5;
}
.xai-info {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 14px;
  background: rgba(111,106,232,0.07);
  border: 1px solid rgba(111,106,232,0.28);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: #a8a3ff;
  line-height: 1.5;
}

/* ── Context strip (sticky page header) ── */
.xai-context-strip {
  position: sticky;
  top: 0;
  z-index: 30;
  display: flex;
  align-items: center;
  gap: 12px;
  height: 54px;
  padding: 0 4px;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
  background: rgba(26,28,40,0.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

/* ── Animations ── */
@keyframes fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0);   }
}
@keyframes slide-up {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0);    }
}
@keyframes skeleton-pulse {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.4; }
}
/* Token-by-token narration reveal (LLM streaming effect, demo flourish).
   Each word span carries its own animation-delay; the card HTML is stable
   across BEV reruns, so this plays once per new narration then holds. */
@keyframes token-in {
  from { opacity: 0; filter: blur(5px); transform: translateY(2px); }
  to   { opacity: 1; filter: blur(0);   transform: translateY(0);   }
}
/* Identical twin of token-in. Alternating the animation-name between the two
   per narration restarts the reveal even when react-markdown patches the span
   nodes in place instead of recreating them. */
@keyframes token-in-2 {
  from { opacity: 0; filter: blur(5px); transform: translateY(2px); }
  to   { opacity: 1; filter: blur(0);   transform: translateY(0);   }
}
@keyframes caret-blink {
  0%,100% { opacity: 1; }
  50%     { opacity: 0; }
}
.xai-fade-in  { animation: fade-in 280ms var(--ease-out) both; }
.xai-slide-up { animation: slide-up 340ms var(--ease-out) both; }
.xai-delay-1  { animation-delay: 60ms  !important; }
.xai-delay-2  { animation-delay: 120ms !important; }
.xai-delay-3  { animation-delay: 180ms !important; }
.xai-skeleton {
  border-radius: 6px;
  background: rgba(111,106,232,0.08);
  animation: skeleton-pulse 1.6s cubic-bezier(0.4,0,0.6,1) infinite;
}

/* ── Interactive card (hover lift + arrow slide) ── */
.panel-interactive {
  transition: transform var(--dur-base) var(--ease-out),
              border-color var(--dur-base) var(--ease-out),
              box-shadow var(--dur-base) var(--ease-out);
}
.panel-interactive:hover {
  transform: translateY(-2px);
  border-color: rgba(111,106,232,0.45);
  box-shadow: 0 14px 34px rgba(0,0,0,0.42);
}
.feature-arrow {
  display: inline-block;
  transition: transform var(--dur-base) var(--ease-out);
}
.panel-interactive:hover .feature-arrow { transform: translateX(4px); }

/* ── Hero eyebrow ── */
.xai-eyebrow {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--primary);
  font-family: var(--font-mono);
}

/* ── Hero ambient glow ── */
.xai-hero { position: relative; }
.xai-hero::before {
  content: "";
  position: absolute;
  left: 50%;
  top: -20px;
  transform: translateX(-50%);
  width: 720px;
  max-width: 92%;
  height: 360px;
  z-index: -1;
  pointer-events: none;
  background: radial-gradient(ellipse at center,
              rgba(111,106,232,0.20),
              rgba(111,106,232,0.06) 45%,
              transparent 72%);
}
"""

# ─── Streamlit component overrides ────────────────────────────────────────────
OVERRIDES_CSS = """
/* ── App shell ── */
.stApp { background: var(--background) !important; }
.main .block-container {
  padding: 0 28px 32px !important;
  max-width: 100% !important;
}
.stApp, .stApp * { font-family: var(--font-sans) !important; }

/* ── Restore Material Symbols Rounded font for Streamlit icons ──────────────
   Streamlit 1.42+ renders ALL Material icons (sidebar collapse button,
   expander toggle, radio indicators, button icons, etc.) as:
       <span data-testid="stIconMaterial" translate="no">icon_name</span>
   The icon_name text becomes a glyph via the "Material Symbols Rounded" font
   (bundled by Streamlit). Our wildcard font-family override breaks this.
   This rule restores the correct font. MUST come after the wildcard above. ── */
[data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded" !important;
  font-weight: 400 !important;
  font-style: normal !important;
  font-feature-settings: "liga" 1 !important;
  -webkit-font-feature-settings: "liga" 1 !important;
  -webkit-font-smoothing: antialiased !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  white-space: nowrap !important;
  word-wrap: normal !important;
  direction: ltr !important;
  line-height: 1 !important;
  display: inline-block !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--sidebar) !important;
  border-right: 1px solid var(--sidebar-border) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] * { color: var(--sidebar-fg) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid var(--sidebar-border) !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] hr { border-color: var(--sidebar-border) !important; }

/* ── Sidebar button nav items ── */
/* Remove margin between consecutive nav buttons */
[data-testid="stSidebar"] [data-testid="stButton"] {
  margin-bottom: 1px !important;
}
/* Base style for ALL nav buttons: transparent, left-aligned text */
[data-testid="stSidebar"] [data-testid^="stBaseButton"] {
  background: transparent !important;
  border: none !important;
  border-radius: 8px !important;
  color: var(--sidebar-fg, #e8e8ef) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 9px 12px !important;
  justify-content: flex-start !important;
  text-align: left !important;
  letter-spacing: -0.01em !important;
  transition: background var(--dur-fast) !important;
  box-shadow: none !important;
  transform: none !important;
  gap: 10px !important;
}
[data-testid="stSidebar"] [data-testid^="stBaseButton"]:hover {
  background: rgba(255,255,255,0.04) !important;
  transform: none !important;
  color: #fff !important;
}
/* Active nav item: type="primary" → data-testid="stBaseButton-primary" */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
  background: rgba(111,106,232,0.15) !important;
  color: var(--sidebar-primary, #6f6ae8) !important;
  font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
  background: rgba(111,106,232,0.22) !important;
  color: var(--sidebar-primary, #6f6ae8) !important;
}
/* Icon size inside nav buttons */
[data-testid="stSidebar"] [data-testid^="stBaseButton"] [data-testid="stIconMaterial"] {
  font-size: 17px !important;
  width: 17px !important;
  height: 17px !important;
  opacity: 0.85 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] [data-testid="stIconMaterial"] {
  opacity: 1 !important;
  color: var(--sidebar-primary, #6f6ae8) !important;
}

/* ── Content-area Radio → toggle buttons ── */
:not([data-testid="stSidebar"]) > [data-testid="stRadio"] > div {
  gap: 4px !important;
}
:not([data-testid="stSidebar"]) > [data-testid="stRadio"] > div > label {
  display: flex !important;
  align-items: center !important;
  gap: 8px !important;
  padding: 8px 12px !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  cursor: pointer !important;
  transition: background var(--dur-fast), border-color var(--dur-fast) !important;
  background: var(--secondary) !important;
  color: var(--foreground) !important;
  width: 100% !important;
}
:not([data-testid="stSidebar"]) > [data-testid="stRadio"] > div > label:hover {
  background: rgba(255,255,255,0.04) !important;
  border-color: rgba(111,106,232,0.4) !important;
}
:not([data-testid="stSidebar"]) > [data-testid="stRadio"] > div > label:has(input:checked) {
  border-color: rgba(111,106,232,0.5) !important;
  background: rgba(111,106,232,0.1) !important;
  color: #a8a3ff !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
  color: var(--foreground) !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
}
h1 { font-size: 28px !important; }
h2 { font-size: 22px !important; }
h3 { font-size: 18px !important; }
p  { color: var(--foreground) !important; }

/* ── st.segmented_control ── */
[data-testid="stSegmentedControl"] {
  background: var(--secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  padding: 3px !important;
  gap: 2px !important;
  display: inline-flex !important;
  margin-bottom: 12px !important;
}
[data-testid="stSegmentedControl"] button {
  border-radius: 999px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 4px 16px !important;
  color: var(--muted-fg) !important;
  background: transparent !important;
  border: none !important;
  transition: all var(--dur-fast) !important;
  min-height: unset !important;
  line-height: 1.6 !important;
}
[data-testid="stSegmentedControl"] button:hover {
  color: var(--foreground) !important;
  background: rgba(255,255,255,0.06) !important;
}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[data-active="true"] {
  background: var(--primary) !important;
  color: #fff !important;
  font-weight: 600 !important;
}

/* ── st.tabs ── */
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  color: var(--muted-fg) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  border-radius: 0 !important;
  transition: color var(--dur-base) var(--ease-out),
              border-color var(--dur-base) var(--ease-out) !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--foreground) !important;
  background: rgba(255,255,255,0.03) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--foreground) !important;
  border-bottom-color: var(--primary) !important;
}

/* ── st.expander ── */
[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-panel) !important;
  overflow: hidden !important;
  margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
  padding: 13px 16px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  color: var(--foreground) !important;
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  letter-spacing: -0.005em !important;
}
[data-testid="stExpander"] summary:hover {
  background: rgba(255,255,255,0.025) !important;
}
[data-testid="stExpander"] summary svg { color: var(--muted-fg) !important; }

/* ── st.metric ── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px !important;
  box-shadow: var(--shadow-panel) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--muted-fg) !important;
}
[data-testid="stMetricValue"] {
  font-size: 22px !important;
  font-weight: 700 !important;
  color: var(--foreground) !important;
  font-variant-numeric: tabular-nums !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--primary) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 9px 18px !important;
  letter-spacing: -0.01em !important;
  transition: background var(--dur-base) var(--ease-out),
              transform  var(--dur-fast) var(--ease-out) !important;
}
.stButton > button:hover  { background: #5d58d4 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0px) !important; }
.stButton > button:disabled { opacity: 0.45 !important; cursor: not-allowed !important; transform: none !important; }

/* ── Inputs / Select ── */
.stSelectbox  > div > div,
.stMultiSelect > div > div {
  background: var(--input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--foreground) !important;
  font-size: 13px !important;
}
.stSelectbox  > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(111,106,232,0.2) !important;
}
.stCheckbox span { color: var(--foreground) !important; }

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
  background: var(--secondary) !important;
  height: 4px !important;
  border-radius: 999px !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--primary) !important;
  border: 2px solid var(--background) !important;
  box-shadow: 0 0 0 3px rgba(111,106,232,0.28) !important;
  width: 16px !important;
  height: 16px !important;
  border-radius: 50% !important;
}

/* ── Alert boxes ── */
[data-testid="stInfo"] {
  background: rgba(111,106,232,0.08) !important;
  border: 1px solid rgba(111,106,232,0.28) !important;
  border-radius: var(--radius) !important;
  color: var(--foreground) !important;
}
[data-testid="stWarning"] {
  background: rgba(245,192,69,0.08) !important;
  border: 1px solid rgba(245,192,69,0.28) !important;
  border-radius: var(--radius) !important;
  color: var(--foreground) !important;
}
[data-testid="stError"] {
  background: rgba(224,82,73,0.08) !important;
  border: 1px solid rgba(224,82,73,0.28) !important;
  border-radius: var(--radius) !important;
  color: var(--foreground) !important;
}
[data-testid="stSuccess"] {
  background: rgba(47,190,135,0.08) !important;
  border: 1px solid rgba(47,190,135,0.28) !important;
  border-radius: var(--radius) !important;
  color: var(--foreground) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-panel) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
  border-radius: 999px !important;
  background: var(--secondary) !important;
  height: 5px !important;
}
.stProgress > div > div > div {
  background: var(--primary) !important;
  border-radius: 999px !important;
  transition: width var(--dur-slow) var(--ease-out) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Column gap ── */
[data-testid="column"] { padding: 0 8px !important; }

/* ── Control rail ── */
.xai-rail [data-testid="stSelectbox"],
.xai-rail [data-testid="stMultiSelect"],
.xai-rail [data-testid="stRadio"],
.xai-rail [data-testid="stSlider"],
.xai-rail [data-testid="stCheckbox"] {
  margin-bottom: 4px !important;
}

/* ── Dataframe ── */
.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}

/* ── Plotly charts ── */
.js-plotly-plot .plotly,
.js-plotly-plot .main-svg {
  border-radius: var(--radius) !important;
}

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaptionContainer"] {
  font-size: 12px !important;
  color: var(--muted-fg) !important;
}

/* ── Home: ghost CTA buttons on cards + secondary hero CTA ──────────────────
   Card navigation and the secondary hero action use real st.button widgets
   keyed with a `home_feat_*` / `home_scen_*` / `home_cta_narr` prefix.
   Streamlit tags each keyed widget container with a `st-key-<key>` class,
   which we target to render them as quiet ghost buttons so they don't compete
   with the solid primary CTA. ── */
div[class*="st-key-home_feat"] .stButton > button,
div[class*="st-key-home_scen"] .stButton > button,
div[class*="st-key-home_cta_narr"] .stButton > button {
  background: transparent !important;
  color: var(--primary) !important;
  border: 1px solid var(--border) !important;
  font-weight: 600 !important;
}
div[class*="st-key-home_feat"] .stButton > button:hover,
div[class*="st-key-home_scen"] .stButton > button:hover,
div[class*="st-key-home_cta_narr"] .stButton > button:hover {
  background: rgba(111,106,232,0.10) !important;
  border-color: rgba(111,106,232,0.45) !important;
  color: #a8a3ff !important;
  transform: none !important;
}
"""

FULL_CSS = TOKENS_CSS + GLOBAL_CSS + UTILITY_CSS + OVERRIDES_CSS


def inject_theme() -> None:
    """Inject the full XAI4AD design system into the current Streamlit page."""
    st.markdown(FONT_LINK, unsafe_allow_html=True)
    st.markdown(f"<style>{FULL_CSS}</style>", unsafe_allow_html=True)
    _apply_mpl_dark()


# ─── Matplotlib dark theme ────────────────────────────────────────────────────

def _apply_mpl_dark() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor":  "#23263a",
        "axes.facecolor":    "#1a1c28",
        "axes.edgecolor":    "#393d57",
        "axes.labelcolor":   "#a4a5b0",
        "axes.labelsize":    9,
        "xtick.color":       "#a4a5b0",
        "ytick.color":       "#a4a5b0",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        "#f5f5f8",
        "grid.color":        "#393d57",
        "grid.linestyle":    "--",
        "grid.alpha":        0.45,
        "legend.facecolor":  "#23263a",
        "legend.edgecolor":  "#393d57",
        "legend.labelcolor": "#f5f5f8",
        "legend.fontsize":   8,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.prop_cycle":   mpl.cycler(color=[
            "#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
        ]),
        "figure.autolayout": True,
        "patch.edgecolor":   "#393d57",
    })


# ─── Plotly shared layout ─────────────────────────────────────────────────────

PLOTLY_LAYOUT: dict = dict(
    paper_bgcolor="#23263a",
    plot_bgcolor="#1a1c28",
    font=dict(family="Inter, system-ui, sans-serif", color="#f5f5f8", size=11),
    xaxis=dict(
        gridcolor="#393d57", gridwidth=1, zeroline=False,
        tickfont=dict(color="#a4a5b0", size=10),
    ),
    yaxis=dict(
        gridcolor="#393d57", gridwidth=1, zeroline=False,
        tickfont=dict(color="#a4a5b0", size=10),
    ),
    margin=dict(l=12, r=12, t=36, b=12),
    hoverlabel=dict(
        bgcolor="#23263a", bordercolor="#393d57",
        font=dict(family="Inter", color="#f5f5f8", size=12),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
        font=dict(color="#a4a5b0", size=10),
    ),
    transition=dict(duration=300, easing="cubic-in-out"),
)

# ─── Colour palettes ─────────────────────────────────────────────────────────

CAT_COLORS: dict[str, str] = {
    "sdc_trajectory": "#4C72B0", "sdc":    "#4C72B0",
    "other_agents":   "#DD8452", "agents": "#DD8452",
    "roadgraph":      "#55A868",
    "traffic_lights": "#C44E52",
    "gps_path":       "#8172B2", "gps":   "#8172B2",
}

METHOD_COLORS: list[str] = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]

AGENT_COLORS: list[str] = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#FFD92F", "#A65628", "#F781BF",
]

TONE_COLORS: dict[str, str] = {
    "detailed":        "#DC2626",
    "brief":           "#10B981",
    "caveat":          "#F59E0B",
    "detailed_caveat": "#7C3AED",
    "detailed-caveat": "#7C3AED",
}

# ─── CBM concept grouping ─────────────────────────────────────────────────────
# Semantic categories for the 15 CBM concepts (registry order preserved within
# each group). Used by the live concept-snapshot bar chart in the CBM tab.

CONCEPT_CATEGORIES: dict[str, list[str]] = {
    "Ego Kinematics": [
        "ego_speed", "ego_acceleration",
    ],
    "Obstacles & Proximity": [
        "dist_nearest_object", "num_objects_within_10m",
        "ttc_lead_vehicle", "lead_vehicle_decelerating",
    ],
    "Traffic & Route": [
        "traffic_light_red", "dist_to_traffic_light", "at_intersection",
        "heading_deviation", "progress_along_route",
    ],
    "Path Geometry": [
        "path_curvature_max", "path_net_heading_change",
        "path_straightness", "heading_to_path_end",
    ],
}

CONCEPT_CATEGORY_COLORS: dict[str, str] = {
    "Ego Kinematics":         "#4C72B0",
    "Obstacles & Proximity":  "#DD8452",
    "Traffic & Route":        "#C44E52",
    "Path Geometry":          "#55A868",
}
