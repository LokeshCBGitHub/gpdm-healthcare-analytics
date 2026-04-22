import base64
import io
import logging
import re
from typing import Dict, List, Tuple, Optional, Any

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger('gpdm.chart')

CHART_COLORS = {
    'primary': '#006BA6',
    'secondary': '#0D1C3D',
    'accent': '#286140',
    'healthy': '#2E8B57',
    'warning': '#E6A817',
    'critical': '#C8102E',
    'info': '#006BA6',
    'preventive': '#0D9488',
    'behavioral': '#6B4C9A',
    'operations': '#34567B',
    'text': '#B0BEC5',
    'text_bright': '#ECEFF1',
    'grid': '#37474F',
    'bg': '#0D1C3D',
    'card_bg': '#1A2744',
}

BAR_PALETTE = [
    '#006BA6', '#286140', '#0D9488', '#6B4C9A', '#34567B',
    '#E6A817', '#C8102E', '#0097A7', '#7B8D6F', '#D4782F',
    '#5C6BC0', '#26A69A', '#AB47BC', '#FF7043', '#42A5F5',
]

DEFAULT_DPI = 150
LABEL_MAX_CHARS = 20
MAX_CHART_WIDTH = 720

def _setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'axes.edgecolor': CHART_COLORS['grid'],
        'axes.labelcolor': CHART_COLORS['text'],
        'xtick.color': CHART_COLORS['text'],
        'ytick.color': CHART_COLORS['text'],
        'text.color': CHART_COLORS['text'],
        'grid.color': CHART_COLORS['grid'],
        'grid.alpha': 0.2,
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })


def _fig_to_base64(fig, dpi=None) -> str:
    if dpi is None:
        dpi = DEFAULT_DPI
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                transparent=True, pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _to_img_tag(b64: str, alt: str = "Chart") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:100%;max-width:{MAX_CHART_WIDTH}px;border-radius:10px;margin:8px 0;" />'


def _get_gradient_colormap(base_color: str):
    try:
        from matplotlib.colors import hex2color, rgb2hex
        rgb = hex2color(base_color)
        dark = tuple(c * 0.6 for c in rgb)
        light = rgb
        return LinearSegmentedColormap.from_list('gradient', [dark, light])
    except (ValueError, IndexError) as e:
        logger.debug('Color operation failed: %s', e)
        return None


def _add_subtle_background(ax):
    try:
        ax.set_facecolor('#0F1B2D')
    except (ValueError, TypeError) as e:
        logger.debug('Chart element skipped: %s', e)


def bar_chart(data: List[Tuple[str, float]], title: str = "",
              color: str = None, max_bars: int = 15,
              currency: bool = False) -> str:
    if not HAS_MATPLOTLIB or not data:
        return ''

    _setup_dark_style()
    data = data[:max_bars]
    labels = [str(d[0])[:LABEL_MAX_CHARS] for d in reversed(data)]
    values = [float(d[1]) for d in reversed(data)]

    fig_height = max(3.5, len(data) * 0.45 + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_height), dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(data))]
    colors.reverse()

    bars = ax.barh(labels, values, color=colors, height=0.65,
                   edgecolor='none', alpha=0.85)

    for i, (label, val) in enumerate(zip(labels, values)):
        ax.barh([label], [val], height=0.65, color='black',
               alpha=0.1, edgecolor='none', zorder=0)

    max_val = max(values) if values else 1
    total_val = sum(values)
    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + max_val * 0.02
        pct = (val / total_val * 100) if total_val > 0 else 0
        label_text = f'{_format_num(val)} ({pct:.1f}%)'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label_text, va='center', fontsize=9,
                color=CHART_COLORS['text_bright'], fontweight='bold')

    ax.set_xlim(0, max_val * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CHART_COLORS['grid'])
    ax.spines['bottom'].set_color(CHART_COLORS['grid'])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _format_num(x)))
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.15, linestyle='--', linewidth=0.5)

    if title:
        ax.set_title(title, pad=15, loc='left', fontsize=14, fontweight='bold')

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), title or "Bar Chart")

def donut_chart(data: Dict[str, int], title: str = "",
                max_slices: int = 8) -> str:
    if not HAS_MATPLOTLIB or not data:
        return ''

    _setup_dark_style()

    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_items) > max_slices:
        main = sorted_items[:max_slices - 1]
        other_total = sum(v for _, v in sorted_items[max_slices - 1:])
        main.append(('Other', other_total))
        sorted_items = main

    labels = [str(k)[:LABEL_MAX_CHARS] for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        pctdistance=0.82, startangle=90,
        wedgeprops={'width': 0.35, 'edgecolor': '#0D1C3D', 'linewidth': 0.5},
    )

    lighter_colors = []
    for c in colors:
        try:
            from matplotlib.colors import hex2color, rgb2hex
            rgb = hex2color(c)
            lighter = tuple(min(1, x * 1.2) for x in rgb)
            lighter_colors.append(rgb2hex(lighter))
        except (ValueError, IndexError) as e:
            logger.debug('Chart element skipped: %s', e)
            lighter_colors.append(c)

    ax.pie(
        values, labels=None, colors=lighter_colors,
        startangle=90,
        wedgeprops={'width': 0.1, 'edgecolor': 'none', 'alpha': 0.3},
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_zorder(10)

    total = sum(values)
    ax.text(0, 0.05, f'{_format_num(total)}', ha='center', va='center',
            fontsize=22, fontweight='bold', color=CHART_COLORS['text_bright'])
    ax.text(0, -0.15, 'total', ha='center', va='center',
            fontsize=10, color=CHART_COLORS['text'], alpha=0.7)

    legend_labels = []
    for label, val in zip(labels, values):
        pct = (val / total * 100) if total > 0 else 0
        legend_labels.append(f'{label} ({pct:.1f}% — {_format_num(val)})')

    legend = ax.legend(wedges, legend_labels, loc='center left',
                       bbox_to_anchor=(1, 0.5), fontsize=9,
                       frameon=False, labelcolor=CHART_COLORS['text'])

    if title:
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold')

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), title or "Donut Chart")

def line_chart(data: List[Tuple[str, float]], title: str = "",
               y_label: str = "", fill: bool = True,
               show_min_max: bool = True) -> str:
    if not HAS_MATPLOTLIB or not data:
        return ''

    _setup_dark_style()

    labels = [str(d[0])[:LABEL_MAX_CHARS] for d in data]
    values = [float(d[1]) for d in data]
    x = list(range(len(values)))

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    ax.plot(x, values, color=CHART_COLORS['primary'], linewidth=3,
            marker='o', markersize=6, markerfacecolor='white',
            markeredgecolor=CHART_COLORS['primary'], markeredgewidth=2.5,
            zorder=5)

    if fill and len(values) > 0:
        ax.fill_between(x, values, alpha=0.25, color=CHART_COLORS['primary'])
        for i in range(len(x) - 1):
            ax.fill_between([x[i], x[i+1]], values[i], values[i+1],
                           alpha=0.1, color=CHART_COLORS['primary'])

    if len(values) >= 3:
        z = np.polyfit(x, values, 1)
        trend = np.poly1d(z)
        ax.plot(x, trend(x), '--', color=CHART_COLORS['warning'],
                linewidth=2, alpha=0.6, label='Trend', zorder=3)

    if show_min_max and len(values) > 1:
        min_idx = values.index(min(values))
        max_idx = values.index(max(values))

        ax.annotate('', xy=(min_idx, values[min_idx]), xytext=(min_idx, values[min_idx] * 0.95),
                   arrowprops=dict(arrowstyle='->', color=CHART_COLORS['critical'], lw=1.5, alpha=0.6))
        ax.text(min_idx, values[min_idx] * 0.92, f"Min: {_format_num(values[min_idx])}",
               ha='center', fontsize=8, color=CHART_COLORS['critical'], alpha=0.8)

        ax.annotate('', xy=(max_idx, values[max_idx]), xytext=(max_idx, values[max_idx] * 1.05),
                   arrowprops=dict(arrowstyle='->', color=CHART_COLORS['healthy'], lw=1.5, alpha=0.6))
        ax.text(max_idx, values[max_idx] * 1.08, f"Max: {_format_num(values[max_idx])}",
               ha='center', fontsize=8, color=CHART_COLORS['healthy'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 8 else 0,
                        ha='right' if len(labels) > 8 else 'center',
                        fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CHART_COLORS['grid'])
    ax.spines['bottom'].set_color(CHART_COLORS['grid'])
    ax.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: _format_num(y)))

    if y_label:
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    if title:
        ax.set_title(title, pad=15, loc='left', fontsize=14, fontweight='bold')

    if len(values) >= 3:
        ax.legend(frameon=False, fontsize=9, loc='upper left')

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), title or "Line Chart")

def grouped_bar_chart(labels: List[str], datasets: Dict[str, List[float]],
                      title: str = "") -> str:
    if not HAS_MATPLOTLIB or not labels or not datasets:
        return ''

    _setup_dark_style()

    n_groups = len(labels)
    n_metrics = len(datasets)
    bar_width = 0.8 / n_metrics
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    for i, (metric_name, values) in enumerate(datasets.items()):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        color = BAR_PALETTE[i % len(BAR_PALETTE)]

        bars = ax.bar(x + offset, values[:n_groups], bar_width * 0.9,
                     label=metric_name, color=color, alpha=0.85,
                     edgecolor=CHART_COLORS['grid'], linewidth=0.5)

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       _format_num(height),
                       ha='center', va='bottom', fontsize=8,
                       color=CHART_COLORS['text_bright'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(l)[:LABEL_MAX_CHARS] for l in labels], rotation=45,
                        ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CHART_COLORS['grid'])
    ax.spines['bottom'].set_color(CHART_COLORS['grid'])
    ax.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: _format_num(y)))

    legend = ax.legend(frameon=False, fontsize=9, loc='upper right',
                      fancybox=False, shadow=False)
    for text in legend.get_texts():
        text.set_color(CHART_COLORS['text'])

    if title:
        ax.set_title(title, pad=15, loc='left', fontsize=14, fontweight='bold')

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), title or "Grouped Bar Chart")

def gauge_chart(value: float, label: str = "", max_val: float = 100) -> str:
    if not HAS_MATPLOTLIB:
        return ''

    _setup_dark_style()

    fig, ax = plt.subplots(figsize=(3.5, 2.5), subplot_kw={'projection': 'polar'}, dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    pct = min(value / max_val, 1.0)
    theta = np.linspace(np.pi, 0, 100)

    low_zone = np.linspace(np.pi, np.pi * 0.67, 33)
    mid_zone = np.linspace(np.pi * 0.67, np.pi * 0.33, 34)
    high_zone = np.linspace(np.pi * 0.33, 0, 33)

    ax.plot(low_zone, [1] * len(low_zone), linewidth=14, color=CHART_COLORS['critical'], alpha=0.2)
    ax.plot(mid_zone, [1] * len(mid_zone), linewidth=14, color=CHART_COLORS['warning'], alpha=0.2)
    ax.plot(high_zone, [1] * len(high_zone), linewidth=14, color=CHART_COLORS['healthy'], alpha=0.2)

    n_fill = int(pct * 100)
    if n_fill > 0:
        if pct >= 0.7:
            color = CHART_COLORS['healthy']
        elif pct >= 0.4:
            color = CHART_COLORS['warning']
        else:
            color = CHART_COLORS['critical']

        ax.plot(theta[:n_fill], [1] * n_fill, linewidth=14, color=color, zorder=10)

    ax.set_ylim(0, 1.5)
    ax.set_axis_off()

    ax.text(np.pi / 2, 0.3, f'{value:.1f}%' if max_val == 100 else _format_num(value),
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=CHART_COLORS['text_bright'])
    if label:
        ax.text(np.pi / 2, -0.15, label, ha='center', va='center',
                fontsize=9, color=CHART_COLORS['text'], alpha=0.8)

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), label or "Gauge")

def stacked_area_chart(data: Dict[str, List[float]], labels: List[str],
                      title: str = "", y_label: str = "") -> str:
    if not HAS_MATPLOTLIB or not data or not labels:
        return ''

    _setup_dark_style()

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    categories = list(data.keys())
    values_array = [data[cat] for cat in categories]
    colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(categories))]

    ax.stackplot(x, *values_array, labels=categories, colors=colors,
                alpha=0.8, edgecolor=CHART_COLORS['grid'], linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l)[:LABEL_MAX_CHARS] for l in labels], rotation=45,
                        ha='right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CHART_COLORS['grid'])
    ax.spines['bottom'].set_color(CHART_COLORS['grid'])
    ax.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: _format_num(y)))

    legend = ax.legend(frameon=False, fontsize=9, loc='upper left',
                      fancybox=False, shadow=False)
    for text in legend.get_texts():
        text.set_color(CHART_COLORS['text'])

    if y_label:
        ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    if title:
        ax.set_title(title, pad=15, loc='left', fontsize=14, fontweight='bold')

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), title or "Stacked Area Chart")

def heatmap_chart(data: np.ndarray, x_labels: List[str],
                 y_labels: List[str], title: str = "",
                 cmap_name: str = 'Blues') -> str:
    if not HAS_MATPLOTLIB or data is None:
        return ''

    _setup_dark_style()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DEFAULT_DPI)
    _add_subtle_background(ax)

    im = ax.imshow(data, cmap=cmap_name, aspect='auto', alpha=0.9)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels([str(l)[:LABEL_MAX_CHARS] for l in x_labels], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([str(l)[:LABEL_MAX_CHARS] for l in y_labels], fontsize=9)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            try:
                val = data[i, j]
                text_color = 'white' if val > (data.max() / 2) else CHART_COLORS['text']
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')
            except (ValueError, TypeError) as e:
                logger.debug('Chart element skipped: %s', e)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Value', rotation=270, labelpad=15, color=CHART_COLORS['text'])
    cbar.ax.tick_params(colors=CHART_COLORS['text'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CHART_COLORS['grid'])
    ax.spines['bottom'].set_color(CHART_COLORS['grid'])

    if title:
        ax.set_title(title, pad=15, loc='left', fontsize=14, fontweight='bold')

    fig.patch.set_facecolor('none')
    return _to_img_tag(_fig_to_base64(fig), title or "Heatmap")

def generate_chart(results: List[Dict], question: str = "",
                   intent: str = "") -> str:
    if not HAS_MATPLOTLIB or not results:
        return ''

    q = question.lower()
    n_rows = len(results)
    cols = list(results[0].keys()) if results else []

    numeric_cols = []
    label_cols = []
    date_cols = []

    for col in cols:
        sample_vals = [r.get(col) for r in results[:20] if r.get(col) is not None]
        if not sample_vals:
            continue

        col_lower = col.lower()
        if any(d in col_lower for d in ['date', 'month', 'year', 'period', 'quarter']):
            date_cols.append(col)
            continue

        is_id_col = any(id_word in col_lower for id_word in [
            '_id', 'npi', 'mrn', 'code', 'name', 'status', 'type',
            'region', 'specialty', 'department', 'facility', 'category',
        ])
        if is_id_col:
            label_cols.append(col)
            continue

        numeric_count = 0
        for v in sample_vals[:10]:
            try:
                float(str(v).replace(',', ''))
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        if numeric_count >= len(sample_vals[:10]) * 0.7:
            numeric_cols.append(col)
        else:
            label_cols.append(col)

    charts_html = ''

    if n_rows == 1 and numeric_cols:
        return ''

    if 'heatmap' in q and label_cols and numeric_cols:
        try:
            lbl_col = label_cols[0]
            numeric_col = numeric_cols[0]
            x_labels = sorted(set(str(r.get(lbl_col, '')) for r in results))[:12]
            y_labels = sorted(set(str(r.get(numeric_col, '')) for r in results))[:12]

            heatmap_data = np.zeros((len(y_labels), len(x_labels)))
            for i, y_label in enumerate(y_labels):
                for j, x_label in enumerate(x_labels):
                    matching_rows = [r for r in results
                                   if str(r.get(lbl_col, '')) == x_label
                                   and str(r.get(numeric_col, '')) == y_label]
                    if matching_rows:
                        heatmap_data[i, j] = float(len(matching_rows))

            charts_html += heatmap_chart(heatmap_data, x_labels, y_labels,
                                        title="Heatmap")
        except (ValueError, TypeError) as e:
            logger.debug('Chart element skipped: %s', e)

    if 'stacked' in q and date_cols and len(numeric_cols) >= 2:
        try:
            date_col = date_cols[0]
            dates = sorted(set(str(r.get(date_col, '')) for r in results))[:20]

            stacked_data = {}
            for nc in numeric_cols[:5]:
                vals = []
                for date in dates:
                    date_rows = [r for r in results if str(r.get(date_col, '')) == date]
                    total = sum(float(str(r.get(nc, 0)).replace(',', '')) for r in date_rows)
                    vals.append(total)
                stacked_data[nc] = vals

            charts_html += stacked_area_chart(stacked_data, dates,
                                             title="Trend by Category")
        except (ValueError, TypeError) as e:
            logger.debug('Chart element skipped: %s', e)

    if (date_cols and numeric_cols) or any(w in q for w in ['trend', 'over time', 'monthly', 'yearly']):
        date_col = date_cols[0] if date_cols else label_cols[0] if label_cols else cols[0]
        val_col = numeric_cols[0] if numeric_cols else cols[-1]
        data = []
        for r in results:
            try:
                data.append((str(r.get(date_col, '')), float(str(r.get(val_col, 0)).replace(',', ''))))
            except (ValueError, TypeError):
                pass
        if data:
            charts_html += line_chart(data, title=f"{val_col} Over Time", y_label=val_col)

    elif any(w in q for w in ['compare', 'versus', ' vs ', 'comparison']) and len(numeric_cols) >= 2 and label_cols:
        lbl = label_cols[0]
        labels = [str(r.get(lbl, ''))[:LABEL_MAX_CHARS] for r in results[:15]]
        datasets = {}
        for nc in numeric_cols[:3]:
            vals = []
            for r in results[:15]:
                try:
                    vals.append(float(str(r.get(nc, 0)).replace(',', '')))
                except (ValueError, TypeError) as e:
                    logger.debug('Chart element skipped: %s', e)
                    vals.append(0)
            datasets[nc] = vals
        charts_html += grouped_bar_chart(labels, datasets,
                                         title=f"Comparison by {lbl}")

    elif any(w in q for w in ['rate', 'percentage', 'pct']) and n_rows <= 10 and numeric_cols:
        val_col = numeric_cols[0]
        lbl_col = label_cols[0] if label_cols else cols[0]
        gauge_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">'
        for r in results[:6]:
            try:
                v = float(str(r.get(val_col, 0)).replace(',', ''))
                lbl = str(r.get(lbl_col, ''))[:LABEL_MAX_CHARS]
                gauge_html += gauge_chart(v, label=lbl)
            except (ValueError, TypeError) as e:
                logger.debug('Chart element skipped: %s', e)
        gauge_html += '</div>'
        charts_html += gauge_html

    elif n_rows <= 6 and label_cols and numeric_cols and \
         any(w in q for w in ['by', 'breakdown', 'distribution', 'status']):
        lbl_col = label_cols[0]
        val_col = numeric_cols[0]
        data = {}
        for r in results:
            try:
                lbl = str(r.get(lbl_col, 'Unknown'))
                val = int(float(str(r.get(val_col, 0)).replace(',', '')))
                data[lbl] = val
            except (ValueError, TypeError) as e:
                logger.debug('Chart element skipped: %s', e)
        if data and len(data) <= 8:
            charts_html += donut_chart(data, title=f"{val_col} by {lbl_col}")

    if not charts_html and label_cols and numeric_cols and n_rows > 1:
        lbl_col = label_cols[0]
        val_col = numeric_cols[-1]
        data = []
        for r in results[:15]:
            try:
                data.append((
                    str(r.get(lbl_col, '')),
                    float(str(r.get(val_col, 0)).replace(',', ''))
                ))
            except (ValueError, TypeError) as e:
                logger.debug('Chart element skipped: %s', e)
        if data:
            charts_html += bar_chart(data, title=f"{val_col} by {lbl_col}")

    return charts_html

def _format_num(v) -> str:
    try:
        v = float(v)
    except (ValueError, TypeError):
        return str(v)

    if abs(v) >= 1_000_000_000:
        return f'{v / 1e9:.1f}B'
    elif abs(v) >= 1_000_000:
        return f'{v / 1e6:.1f}M'
    elif abs(v) >= 10_000:
        return f'{v / 1e3:.1f}K'
    elif abs(v) >= 1000:
        return f'{v:,.0f}'
    elif v == int(v):
        return str(int(v))
    else:
        return f'{v:.2f}'


if __name__ == '__main__':
    print(f"matplotlib available: {HAS_MATPLOTLIB}")

    if HAS_MATPLOTLIB:
        data = [('Region A', 1500), ('Region B', 1200), ('Region C', 900),
                ('Region D', 600), ('Region E', 300)]
        html = bar_chart(data, title="Claims by Region")
        print(f"Bar chart: {len(html)} chars HTML")

        dist = {'Approved': 8500, 'Denied': 1200, 'Pending': 800, 'Review': 500}
        html = donut_chart(dist, title="Claim Status Distribution")
        print(f"Donut chart: {len(html)} chars HTML")

        trend_data = [(f'2024-{m:02d}', 1000 + m * 100 + (m % 3) * 50)
                      for m in range(1, 13)]
        html = line_chart(trend_data, title="Monthly Claims Trend", y_label="Count")
        print(f"Line chart: {len(html)} chars HTML")

        labels = ['East', 'West', 'North', 'South']
        datasets = {'Billed': [50000, 42000, 38000, 31000],
                    'Paid': [45000, 38000, 35000, 28000]}
        html = grouped_bar_chart(labels, datasets, title="Billed vs Paid by Region")
        print(f"Grouped bar chart: {len(html)} chars HTML")

        html = gauge_chart(72.5, label="Approval Rate")
        print(f"Gauge chart: {len(html)} chars HTML")

        stacked_data = {
            'Category A': [100, 150, 120, 180, 160],
            'Category B': [80, 90, 110, 95, 130],
            'Category C': [60, 70, 85, 75, 90]
        }
        html = stacked_area_chart(stacked_data, ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                                 title="Trend by Category", y_label="Count")
        print(f"Stacked area chart: {len(html)} chars HTML")

        heatmap_data = np.random.rand(5, 5)
        html = heatmap_chart(heatmap_data, ['A', 'B', 'C', 'D', 'E'],
                            ['1', '2', '3', '4', '5'], title="Correlation Matrix")
        print(f"Heatmap chart: {len(html)} chars HTML")

        results = [
            {'KP_REGION': 'East', 'total_billed': '50000', 'total_paid': '45000'},
            {'KP_REGION': 'West', 'total_billed': '42000', 'total_paid': '38000'},
            {'KP_REGION': 'North', 'total_billed': '38000', 'total_paid': '35000'},
        ]
        html = generate_chart(results, "compare billed vs paid by region", "comparison")
        print(f"Auto chart (comparison): {len(html)} chars HTML")

        print("\nAll chart types generated successfully!")
