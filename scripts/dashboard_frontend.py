from typing import Dict, Any, List, Optional
import json


class DashboardFrontendRenderer:

    KP_BLUE = "#002B5C"
    KP_LIGHT = "#F5F7FA"
    KP_WHITE = "#FFFFFF"

    RAG_GREEN = "#28a745"
    RAG_YELLOW = "#ffc107"
    RAG_RED = "#dc3545"

    CHART_COLORS = [
        "#002B5C", "#0052B3", "#0066E6", "#4D94FF",
        "#80B3FF", "#B3D9FF", "#FF6B6B", "#4ECDC4"
    ]

    def __init__(self, db_path: str = None):
        self.chart_counter = 0
        self.insights_engine = None
        if db_path:
            try:
                from kpi_insights_engine import KPIInsightsEngine
                self.insights_engine = KPIInsightsEngine(db_path)
            except Exception:
                pass

    def _get_chart_id(self) -> str:
        self.chart_counter += 1
        return f"chart_{self.chart_counter}"

    def _get_base_html(self, title: str, content: str, scripts: str = "") -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                         'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            background-color: {self.KP_LIGHT};
            color: #333;
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, {self.KP_BLUE} 0%, #003d7a 100%);
            color: {self.KP_WHITE};
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 43, 92, 0.1);
        }}

        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }}

        .logo-placeholder {{
            width: 50px;
            height: 50px;
            background: {self.KP_WHITE};
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: {self.KP_BLUE};
            font-size: 24px;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .grid {{
            display: grid;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .grid-2 {{
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }}

        .grid-3 {{
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }}

        .grid-6 {{
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        }}

        .card {{
            background: {self.KP_WHITE};
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
        }}

        .card:hover {{
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
            transform: translateY(-2px);
        }}

        .card-full {{
            grid-column: 1 / -1;
        }}

        .metric-card {{
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {self.KP_BLUE};
            margin-bottom: 0.5rem;
        }}

        .metric-label {{
            font-size: 0.875rem;
            color: #666;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.75rem;
        }}

        .metric-change {{
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }}

        .metric-change.positive {{
            color: {self.RAG_GREEN};
        }}

        .metric-change.negative {{
            color: {self.RAG_RED};
        }}

        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 1rem;
        }}

        .chart-container-large {{
            position: relative;
            height: 400px;
            margin-bottom: 1rem;
        }}

        .chart-title {{
            font-size: 1.125rem;
            font-weight: 600;
            color: {self.KP_BLUE};
            margin-bottom: 1rem;
            margin-top: 1.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        thead {{
            background-color: {self.KP_LIGHT};
        }}

        th {{
            text-align: left;
            padding: 1rem;
            font-weight: 600;
            color: {self.KP_BLUE};
            border-bottom: 2px solid {self.KP_BLUE};
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 0.875rem 1rem;
            border-bottom: 1px solid #e0e0e0;
        }}

        tbody tr:hover {{
            background-color: {self.KP_LIGHT};
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .status-badge.green {{
            background-color: rgba(40, 167, 69, 0.1);
            color: {self.RAG_GREEN};
        }}

        .status-badge.yellow {{
            background-color: rgba(255, 193, 7, 0.1);
            color: #856404;
        }}

        .status-badge.red {{
            background-color: rgba(220, 53, 69, 0.1);
            color: {self.RAG_RED};
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        .status-dot.green {{
            background-color: {self.RAG_GREEN};
        }}

        .status-dot.yellow {{
            background-color: {self.RAG_YELLOW};
        }}

        .status-dot.red {{
            background-color: {self.RAG_RED};
        }}

        .stars {{
            font-size: 2rem;
            color: #ffc107;
            margin: 1rem 0;
        }}

        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {self.KP_BLUE};
            margin: 2rem 0 1rem 0;
            border-bottom: 3px solid {self.KP_BLUE};
            padding-bottom: 0.75rem;
        }}

        .heatmap-table {{
            width: 100%;
        }}

        .heatmap-cell {{
            text-align: center;
            font-weight: 500;
            padding: 1rem;
            border-radius: 4px;
            color: {self.KP_WHITE};
        }}

        .heatmap-cell.high {{
            background-color: {self.RAG_GREEN};
        }}

        .heatmap-cell.medium {{
            background-color: {self.RAG_YELLOW};
            color: #333;
        }}

        .heatmap-cell.low {{
            background-color: {self.RAG_RED};
        }}

        .progress-bar {{
            width: 100%;
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {self.KP_BLUE}, #0066E6);
            border-radius: 4px;
            transition: width 0.3s ease;
        }}

        .action-item {{
            background-color: {self.KP_LIGHT};
            border-left: 4px solid {self.KP_BLUE};
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 4px;
        }}

        .action-item-title {{
            font-weight: 600;
            color: {self.KP_BLUE};
            margin-bottom: 0.25rem;
        }}

        .action-item-desc {{
            font-size: 0.875rem;
            color: #666;
        }}

        /* Clickable elements */
        .clickable {{
            cursor: pointer;
            position: relative;
        }}

        .clickable::after {{
            content: 'Click for deep insights';
            position: absolute;
            bottom: -4px;
            right: 8px;
            font-size: 0.65rem;
            color: #999;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .clickable:hover::after {{
            opacity: 1;
        }}

        .clickable:hover {{
            box-shadow: 0 4px 16px rgba(0, 43, 92, 0.2) !important;
            transform: translateY(-3px) !important;
        }}

        /* Insight Modal */
        .insight-overlay {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 43, 92, 0.6);
            z-index: 9999;
            overflow-y: auto;
            padding: 2rem;
            backdrop-filter: blur(4px);
        }}

        .insight-overlay.active {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }}

        .insight-modal {{
            background: {self.KP_WHITE};
            border-radius: 12px;
            max-width: 1000px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease;
        }}

        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .insight-header {{
            background: linear-gradient(135deg, {self.KP_BLUE} 0%, #003d7a 100%);
            color: {self.KP_WHITE};
            padding: 1.5rem 2rem;
            border-radius: 12px 12px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .insight-header h2 {{
            font-size: 1.25rem;
            font-weight: 600;
        }}

        .insight-close {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
            border-radius: 50%;
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .insight-close:hover {{
            background: rgba(255,255,255,0.4);
        }}

        .insight-body {{
            padding: 2rem;
        }}

        .insight-section {{
            margin-bottom: 2rem;
        }}

        .insight-section-title {{
            font-size: 1rem;
            font-weight: 700;
            color: {self.KP_BLUE};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {self.KP_BLUE};
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .insight-text {{
            font-size: 0.95rem;
            line-height: 1.7;
            color: #333;
            margin-bottom: 0.75rem;
        }}

        .model-card {{
            background: {self.KP_LIGHT};
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            border-left: 4px solid {self.KP_BLUE};
        }}

        .model-name {{
            font-weight: 700;
            color: {self.KP_BLUE};
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }}

        .model-detail {{
            font-size: 0.85rem;
            color: #555;
            margin-bottom: 0.25rem;
        }}

        .model-accuracy {{
            font-size: 0.8rem;
            color: {self.RAG_GREEN};
            font-weight: 600;
        }}

        .factor-bar {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }}

        .factor-label {{
            min-width: 200px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #333;
        }}

        .factor-bar-track {{
            flex: 1;
            height: 24px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}

        .factor-bar-fill {{
            height: 100%;
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding-left: 8px;
            font-size: 0.75rem;
            color: white;
            font-weight: 600;
        }}

        .factor-detail {{
            font-size: 0.8rem;
            color: #666;
            margin-left: 200px;
            margin-bottom: 0.5rem;
            padding-left: 1rem;
            border-left: 2px solid #ddd;
        }}

        .rec-card {{
            background: {self.KP_WHITE};
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.2s;
        }}

        .rec-card:hover {{
            border-color: {self.KP_BLUE};
            box-shadow: 0 2px 8px rgba(0,43,92,0.1);
        }}

        .rec-priority {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }}

        .rec-priority.critical {{
            background: rgba(220,53,69,0.1);
            color: {self.RAG_RED};
        }}

        .rec-priority.high {{
            background: rgba(255,193,7,0.15);
            color: #856404;
        }}

        .rec-priority.medium {{
            background: rgba(40,167,69,0.1);
            color: {self.RAG_GREEN};
        }}

        .rec-title {{
            font-weight: 700;
            color: {self.KP_BLUE};
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }}

        .rec-detail {{
            font-size: 0.85rem;
            color: #555;
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }}

        .rec-impact {{
            font-size: 0.85rem;
            color: {self.RAG_GREEN};
            font-weight: 600;
        }}

        .rec-timeline {{
            font-size: 0.8rem;
            color: #888;
        }}

        .forecast-chart-container {{
            position: relative;
            height: 250px;
            margin: 1rem 0;
        }}

        .insight-kpi-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {self.KP_BLUE};
            text-align: center;
            margin: 1rem 0 0.25rem;
        }}

        .insight-kpi-label {{
            text-align: center;
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 1rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}

            .header {{
                padding: 1.5rem;
            }}

            .header-content {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .metric-value {{
                font-size: 1.75rem;
            }}

            .grid-2, .grid-3, .grid-6 {{
                grid-template-columns: 1fr;
            }}

            .insight-overlay {{
                padding: 0.5rem;
            }}

            .factor-label {{
                min-width: 120px;
            }}

            .factor-detail {{
                margin-left: 0;
            }}
        }}

        /* ═══════════════════════════════════════════════════ */
        /*  MODERN TOOLBAR                                     */
        /* ═══════════════════════════════════════════════════ */
        .toolbar {{
            background: {self.KP_WHITE};
            border-bottom: 1px solid #e0e0e0;
            padding: 0.75rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .toolbar-left {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .toolbar-right {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .toolbar-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.5rem 1rem;
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            background: {self.KP_WHITE};
            color: #344054;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
        }}
        .toolbar-btn:hover {{
            background: {self.KP_LIGHT};
            border-color: {self.KP_BLUE};
            color: {self.KP_BLUE};
            box-shadow: 0 1px 3px rgba(0,43,92,0.1);
        }}
        .toolbar-btn.primary {{
            background: {self.KP_BLUE};
            color: {self.KP_WHITE};
            border-color: {self.KP_BLUE};
        }}
        .toolbar-btn.primary:hover {{
            background: #003d7a;
            box-shadow: 0 2px 6px rgba(0,43,92,0.25);
        }}
        .toolbar-btn svg {{
            width: 16px;
            height: 16px;
            flex-shrink: 0;
        }}
        .toolbar-divider {{
            width: 1px;
            height: 24px;
            background: #e0e0e0;
            margin: 0 0.25rem;
        }}
        .toolbar-timestamp {{
            font-size: 0.75rem;
            color: #888;
            margin-left: 0.5rem;
        }}

        /* ═══════════════════════════════════════════════════ */
        /*  AI CHATBOT PANEL                                   */
        /* ═══════════════════════════════════════════════════ */
        .chatbot-fab {{
            position: fixed;
            bottom: 28px;
            right: 28px;
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: linear-gradient(135deg, #0052B3 0%, {self.KP_BLUE} 100%);
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: 0 6px 20px rgba(0,43,92,0.35);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            animation: chatPulse 3s infinite;
        }}
        .chatbot-fab:hover {{
            transform: scale(1.1);
            box-shadow: 0 8px 28px rgba(0,43,92,0.5);
        }}
        .chatbot-fab svg {{
            width: 32px;
            height: 32px;
        }}
        .chatbot-fab .fab-badge {{
            position: absolute;
            top: -2px;
            right: -2px;
            width: 18px;
            height: 18px;
            background: {self.RAG_RED};
            border-radius: 50%;
            font-size: 10px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid white;
        }}
        @keyframes chatPulse {{
            0%,100% {{ box-shadow: 0 6px 20px rgba(0,43,92,0.35); }}
            50% {{ box-shadow: 0 6px 28px rgba(0,82,179,0.5); }}
        }}

        .chatbot-panel {{
            position: fixed;
            bottom: 104px;
            right: 28px;
            width: 420px;
            max-height: 600px;
            background: {self.KP_WHITE};
            border-radius: 16px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.18);
            z-index: 10001;
            display: none;
            flex-direction: column;
            overflow: hidden;
            animation: chatSlideUp 0.3s ease;
        }}
        .chatbot-panel.open {{
            display: flex;
        }}
        @keyframes chatSlideUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .chat-header {{
            background: linear-gradient(135deg, {self.KP_BLUE} 0%, #003d7a 100%);
            color: white;
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .chat-header-left {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .chat-avatar {{
            width: 36px;
            height: 36px;
            background: rgba(255,255,255,0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .chat-header-title {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .chat-header-subtitle {{
            font-size: 0.7rem;
            opacity: 0.8;
        }}
        .chat-close {{
            background: rgba(255,255,255,0.15);
            border: none;
            color: white;
            width: 32px;
            height: 32px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .chat-close:hover {{ background: rgba(255,255,255,0.3); }}

        .chat-suggestions {{
            padding: 0.75rem 1rem;
            background: {self.KP_LIGHT};
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }}
        .chat-suggestion {{
            padding: 0.3rem 0.7rem;
            background: white;
            border: 1px solid #d0d5dd;
            border-radius: 16px;
            font-size: 0.72rem;
            color: #344054;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
        }}
        .chat-suggestion:hover {{
            background: {self.KP_BLUE};
            color: white;
            border-color: {self.KP_BLUE};
        }}

        .chat-messages {{
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            min-height: 280px;
            max-height: 360px;
        }}
        .chat-msg {{
            margin-bottom: 1rem;
            display: flex;
            gap: 0.5rem;
        }}
        .chat-msg.user {{
            flex-direction: row-reverse;
        }}
        .chat-msg-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 0.75rem;
        }}
        .bot {{
            /* Bot message styling marker */
        }}
        .user {{
            /* User message styling marker */
        }}
        .chat-msg.bot .chat-msg-avatar {{
            background: linear-gradient(135deg, #0052B3, {self.KP_BLUE});
            color: white;
        }}
        .chat-msg.user .chat-msg-avatar {{
            background: #e0e7ff;
            color: {self.KP_BLUE};
        }}
        .chat-msg-bubble {{
            max-width: 80%;
            padding: 0.65rem 0.9rem;
            border-radius: 12px;
            font-size: 0.85rem;
            line-height: 1.5;
        }}
        .chat-msg.bot .chat-msg-bubble {{
            background: {self.KP_LIGHT};
            color: #333;
            border-bottom-left-radius: 4px;
        }}
        .chat-msg.user .chat-msg-bubble {{
            background: {self.KP_BLUE};
            color: white;
            border-bottom-right-radius: 4px;
        }}
        .chat-msg-bubble table {{
            font-size: 0.78rem;
            margin-top: 0.5rem;
        }}
        .chat-msg-bubble table th {{
            font-size: 0.72rem;
            padding: 0.4rem 0.6rem;
        }}
        .chat-msg-bubble table td {{
            padding: 0.35rem 0.6rem;
        }}
        .chat-typing {{
            display: flex;
            gap: 4px;
            padding: 0.5rem;
        }}
        .chat-typing span {{
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            animation: typingDot 1.4s infinite;
        }}
        .chat-typing span:nth-child(2) {{ animation-delay: 0.2s; }}
        .chat-typing span:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes typingDot {{
            0%,60%,100% {{ transform: translateY(0); opacity: 0.4; }}
            30% {{ transform: translateY(-6px); opacity: 1; }}
        }}

        .chat-input-area {{
            padding: 0.75rem 1rem;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 0.5rem;
            background: white;
        }}
        .chat-input {{
            flex: 1;
            border: 1px solid #d0d5dd;
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            font-size: 0.85rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }}
        .chat-input:focus {{
            border-color: {self.KP_BLUE};
            box-shadow: 0 0 0 3px rgba(0,43,92,0.1);
        }}
        .chat-send {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: {self.KP_BLUE};
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}
        .chat-send:hover {{
            background: #003d7a;
        }}
        .chat-send svg {{
            width: 18px;
            height: 18px;
        }}

        /* Anticipation tooltip */
        .anticipation-bar {{
            position: fixed;
            bottom: 100px;
            right: 100px;
            background: {self.KP_WHITE};
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 9998;
            max-width: 280px;
            font-size: 0.78rem;
            color: #555;
            display: none;
            animation: chatSlideUp 0.3s ease;
        }}
        .anticipation-bar.show {{
            display: block;
        }}
        .anticipation-bar strong {{
            color: {self.KP_BLUE};
        }}

    </style>
</head>
<body>
    
    <div class="toolbar">
        <div class="toolbar-left">
            <button class="toolbar-btn" onclick="window.print();" title="Print dashboard">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9V2h12v7M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
                Print
            </button>
            <button class="toolbar-btn" onclick="exportDashboard('pdf')" title="Export as PDF">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                Export PDF
            </button>
            <button class="toolbar-btn" onclick="exportDashboard('csv')" title="Export data as CSV">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Export CSV
            </button>
            <div class="toolbar-divider"></div>
            <button class="toolbar-btn" onclick="saveDashboardState()" title="Save current view">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
                Save View
            </button>
            <button class="toolbar-btn" onclick="toggleBookmark()" id="bookmarkBtn" title="Bookmark this dashboard">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/></svg>
                Bookmark
            </button>
        </div>
        <div class="toolbar-right">
            <span class="toolbar-timestamp" id="lastUpdated"></span>
            <div class="toolbar-divider"></div>
            <button class="toolbar-btn" onclick="refreshDashboard()" title="Refresh data">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>
                Refresh
            </button>
            <button class="toolbar-btn primary" onclick="toggleChat()" title="Ask the AI analyst">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                Ask AI
            </button>
        </div>
    </div>

    <div class="header">
        <div class="header-content">
            <div class="logo-placeholder">KP</div>
            <h1>{title}</h1>
        </div>
    </div>

    <div class="container">
        {content}
    </div>

    
    <button class="chatbot-fab" onclick="toggleChat()" id="chatFab" title="Ask the AI Healthcare Analyst">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 21a9 9 0 110-18 9 9 0 010 18z" fill="rgba(255,255,255,0.15)"/>
            <path d="M8 10h.01M12 10h.01M16 10h.01"/>
            <path d="M9 16c1 1 2.5 1.5 3 1.5s2-.5 3-1.5"/>
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="none"/>
            <circle cx="8.5" cy="10" r="1" fill="white"/>
            <circle cx="15.5" cy="10" r="1" fill="white"/>
            <path d="M9 15c.83.67 2 1 3 1s2.17-.33 3-1" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M12 2v1M22 12h-1M12 22v-1M2 12h1" stroke="rgba(255,255,255,0.4)" stroke-width="1"/>
            <path d="M4.93 4.93l.7.7M18.36 4.93l-.7.7M18.36 19.07l-.7-.7M4.93 19.07l.7-.7" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
        </svg>
        <div class="fab-badge" id="chatBadge" style="display:none;">!</div>
    </button>

    
    <div class="chatbot-panel" id="chatPanel">
        <div class="chat-header">
            <div class="chat-header-left">
                <div class="chat-avatar">
                    <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" width="20" height="20">
                        <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/>
                        <circle cx="12" cy="7" r="4"/>
                    </svg>
                </div>
                <div>
                    <div class="chat-header-title">KP Healthcare AI Analyst</div>
                    <div class="chat-header-subtitle">Powered by ML &bull; Deep Data Insights</div>
                </div>
            </div>
            <button class="chat-close" onclick="toggleChat()">&times;</button>
        </div>
        <div class="chat-suggestions" id="chatSuggestions">
            <span class="chat-suggestion" onclick="askSuggestion(this)">What is our current MLR?</span>
            <span class="chat-suggestion" onclick="askSuggestion(this)">Show denial rates by region</span>
            <span class="chat-suggestion" onclick="askSuggestion(this)">Risk score distribution</span>
            <span class="chat-suggestion" onclick="askSuggestion(this)">Revenue forecast next 6 months</span>
            <span class="chat-suggestion" onclick="askSuggestion(this)">Member retention trends</span>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="chat-msg bot">
                <div class="chat-msg-avatar">AI</div>
                <div class="chat-msg-bubble">
                    Welcome! I'm your <strong>KP Healthcare AI Analyst</strong>. I have deep understanding of all your dashboard data — financial KPIs, Stars quality measures, member demographics, claims, and more.<br><br>
                    Ask me anything about the data, and I'll provide insights backed by ML models, statistical analysis, and actionable recommendations. What would you like to explore?
                </div>
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" class="chat-input" id="chatInput" placeholder="Ask about KPIs, trends, forecasts..." onkeydown="if(event.key==='Enter')sendChat()">
            <button class="chat-send" onclick="sendChat()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            </button>
        </div>
    </div>

    
    <div class="anticipation-bar" id="anticipationBar"></div>

    
    <div id="insightOverlay" class="insight-overlay" onclick="if(event.target===this)closeInsight();">
        <div id="insightModal" class="insight-modal">
            <div class="insight-header">
                <h2 id="insightTitle">KPI Deep Insights</h2>
                <button class="insight-close" onclick="closeInsight();">&times;</button>
            </div>
            <div class="insight-body" id="insightBody">
            </div>
        </div>
    </div>

    <script>
    // ── Insight Modal Engine (MUST load before page scripts) ──
    var _insightData = {{}};
    var _forecastCharts = {{}};

    function registerInsight(key, data) {{
        _insightData[key] = data;
    }}

    function openInsight(key) {{
        var d = _insightData[key];
        if (!d) return;
        document.getElementById('insightTitle').textContent = d.kpi_name || key;
        document.getElementById('insightBody').innerHTML = buildInsightHTML(d, key);
        document.getElementById('insightOverlay').classList.add('active');
        document.body.style.overflow = 'hidden';
        // Render forecast chart after DOM update
        setTimeout(function() {{ renderForecastChart(d, key); }}, 100);
    }}

    function closeInsight() {{
        document.getElementById('insightOverlay').classList.remove('active');
        document.body.style.overflow = '';
        // Destroy forecast charts
        Object.keys(_forecastCharts).forEach(function(k) {{
            if (_forecastCharts[k]) _forecastCharts[k].destroy();
            delete _forecastCharts[k];
        }});
    }}

    // Keyboard: Escape closes modal
    document.addEventListener('keydown', function(e) {{ if(e.key === 'Escape') closeInsight(); }});

    function buildInsightHTML(d, key) {{
        var html = '';

        // Current value banner
        if (d.current_value !== null && d.current_value !== undefined) {{
            var unit = '';
            if (d.kpi_name && (d.kpi_name.indexOf('Rate') >= 0 || d.kpi_name.indexOf('MLR') >= 0 || d.kpi_name.indexOf('Ratio') >= 0)) unit = '%';
            else if (d.kpi_name && d.kpi_name.indexOf('PMPM') >= 0) unit = '';
            html += '<div class="insight-kpi-value">' + (unit === '' && d.kpi_name && d.kpi_name.indexOf('Risk') < 0 ? '$' : '') + (typeof d.current_value === 'number' ? d.current_value.toLocaleString(undefined, {{maximumFractionDigits:2}}) : d.current_value) + unit + '</div>';
            if (d.target) html += '<div class="insight-kpi-label">Target: ' + d.target + (unit||'') + (d.benchmark ? ' | Benchmark: ' + d.benchmark : '') + '</div>';
        }}

        // 1. What was learned
        if (d.data_learned) {{
            html += '<div class="insight-section"><div class="insight-section-title">What We Learned from the Data</div>';
            html += '<div class="insight-text">' + (d.data_learned.summary || '') + '</div>';
            if (d.data_learned.variance_decomposition) {{
                var vd = d.data_learned.variance_decomposition;
                html += '<div class="insight-text"><strong>Variance Decomposition:</strong> Trend explains ' + (vd.trend_pct||0).toFixed(1) + '%, Seasonal ' + (vd.seasonal_pct||0).toFixed(1) + '%, Residual ' + (vd.residual_pct||0).toFixed(1) + '%</div>';
            }}
            if (d.data_learned.regional_spread) {{
                html += '<div class="insight-text"><strong>Regional Performance:</strong></div><table style="font-size:0.85rem;"><thead><tr><th>Region</th><th>Value</th></tr></thead><tbody>';
                d.data_learned.regional_spread.forEach(function(r) {{
                    var v = r.pmpm || r.mlr || r.avg_risk || r.retention || 0;
                    html += '<tr><td>' + r.region + '</td><td>' + (typeof v === 'number' ? v.toFixed(2) : v) + '</td></tr>';
                }});
                html += '</tbody></table>';
            }}
            if (d.data_learned.denial_breakdown) {{
                html += '<div class="insight-text"><strong>Denial Root Causes:</strong></div><table style="font-size:0.85rem;"><thead><tr><th>Reason</th><th>Count</th><th>$ at Risk</th></tr></thead><tbody>';
                d.data_learned.denial_breakdown.forEach(function(dd) {{
                    html += '<tr><td>' + dd.reason + '</td><td>' + dd.count + '</td><td>$' + (dd.revenue||0).toLocaleString(undefined,{{maximumFractionDigits:0}}) + '</td></tr>';
                }});
                html += '</tbody></table>';
            }}
            if (d.risk_distribution) {{
                html += '<div class="insight-text"><strong>Risk Score Distribution:</strong></div><table style="font-size:0.85rem;"><thead><tr><th>Tier</th><th>Members</th><th>Avg Risk</th></tr></thead><tbody>';
                d.risk_distribution.forEach(function(r) {{
                    html += '<tr><td>' + r.tier + '</td><td>' + r.count.toLocaleString() + '</td><td>' + r.avg.toFixed(3) + '</td></tr>';
                }});
                html += '</tbody></table>';
            }}
            html += '</div>';
        }}

        // 2. Forecast chart
        if (d.historical_values && d.historical_months) {{
            html += '<div class="insight-section"><div class="insight-section-title">AI Forecast (6-Month Projection)</div>';
            html += '<div class="forecast-chart-container"><canvas id="forecastChart_' + key + '"></canvas></div>';
            if (d.forecast_accuracy_rationale) {{
                html += '<div class="insight-text" style="font-style:italic;color:#666;">' + d.forecast_accuracy_rationale + '</div>';
            }}
            html += '</div>';
        }}

        // 3. Models Used
        if (d.models_used && d.models_used.length > 0) {{
            html += '<div class="insight-section"><div class="insight-section-title">ML/AI Models Used</div>';
            d.models_used.forEach(function(m) {{
                html += '<div class="model-card">';
                html += '<div class="model-name">' + m.name + '</div>';
                html += '<div class="model-detail"><strong>Purpose:</strong> ' + (m.purpose || m.why_chosen || '') + '</div>';
                html += '<div class="model-accuracy">Accuracy: ' + (m.accuracy || 'N/A') + '</div>';
                if (m.why_chosen && m.purpose) html += '<div class="model-detail" style="margin-top:0.5rem;">' + m.why_chosen + '</div>';
                if (m.forecast_6m) {{
                    if (Array.isArray(m.forecast_6m)) {{
                        html += '<div class="model-detail"><strong>6-Month Forecast:</strong> ' + m.forecast_6m.map(function(v){{ return typeof v === 'number' ? v.toFixed(2) : v; }}).join(' → ') + '</div>';
                    }} else if (typeof m.forecast_6m === 'object') {{
                        Object.keys(m.forecast_6m).forEach(function(fk) {{
                            var fv = m.forecast_6m[fk];
                            html += '<div class="model-detail"><strong>' + fk + ':</strong> ' + (Array.isArray(fv) ? fv.map(function(v){{return typeof v==='number'?v.toFixed(2):v;}}).join(' → ') : fv) + '</div>';
                        }});
                    }}
                }}
                html += '</div>';
            }});
            html += '</div>';
        }}

        // 4. Contributing Factors
        if (d.contributing_factors && d.contributing_factors.length > 0) {{
            html += '<div class="insight-section"><div class="insight-section-title">Contributing Factors (Ranked by Importance)</div>';
            d.contributing_factors.forEach(function(f) {{
                var pct = ((f.importance || 0) * 100).toFixed(0);
                var color = f.direction === 'positive' || f.direction === 'positive_opportunity' ? '{self.RAG_GREEN}' : f.direction === 'negative' ? '{self.RAG_RED}' : '{self.RAG_YELLOW}';
                html += '<div class="factor-bar"><div class="factor-label">' + f.factor + '</div>';
                html += '<div class="factor-bar-track"><div class="factor-bar-fill" style="width:' + pct + '%;background:' + color + ';">' + pct + '%</div></div></div>';
                html += '<div class="factor-detail">' + (f.detail || '') + '</div>';
            }});
            html += '</div>';
        }}

        // 5. Recommendations
        if (d.recommendations && d.recommendations.length > 0) {{
            html += '<div class="insight-section"><div class="insight-section-title">Detailed Recommendations</div>';
            d.recommendations.forEach(function(r) {{
                var prClass = r.priority === 'critical' ? 'critical' : r.priority === 'high' ? 'high' : 'medium';
                html += '<div class="rec-card">';
                html += '<span class="rec-priority ' + prClass + '">' + (r.priority || 'medium') + '</span>';
                html += '<div class="rec-title">' + r.title + '</div>';
                html += '<div class="rec-detail">' + r.detail + '</div>';
                if (r.impact) html += '<div class="rec-impact">Impact: ' + r.impact + '</div>';
                if (r.timeline) html += '<div class="rec-timeline">Timeline: ' + r.timeline + '</div>';
                html += '</div>';
            }});
            html += '</div>';
        }}

        return html;
    }}

    function renderForecastChart(d, key) {{
        var canvasId = 'forecastChart_' + key;
        var canvas = document.getElementById(canvasId);
        if (!canvas || !d.historical_values || !d.historical_months) return;

        var labels = (d.historical_months || []).concat(d.forecast_months || []);
        var historical = d.historical_values || [];
        var forecastVals = [];

        // Get best forecast (Holt-Winters if available)
        if (d.models_used) {{
            for (var i = 0; i < d.models_used.length; i++) {{
                var m = d.models_used[i];
                if (m.forecast_6m && Array.isArray(m.forecast_6m)) {{
                    forecastVals = m.forecast_6m;
                    break;
                }}
            }}
        }}

        // Build datasets
        var histData = historical.concat(new Array(forecastVals.length).fill(null));
        var fcData = new Array(historical.length - 1).fill(null).concat([historical[historical.length-1]]).concat(forecastVals);

        // Monte Carlo bands
        var p10 = null, p90 = null;
        if (d.models_used) {{
            d.models_used.forEach(function(m) {{
                if (m.name && m.name.indexOf('Monte Carlo') >= 0 && m.forecast_6m && typeof m.forecast_6m === 'object' && !Array.isArray(m.forecast_6m)) {{
                    var keys = Object.keys(m.forecast_6m);
                    keys.forEach(function(k) {{
                        if (k.indexOf('p10') >= 0 || k.indexOf('pessimistic') >= 0) p10 = m.forecast_6m[k];
                        if (k.indexOf('p90') >= 0 || k.indexOf('optimistic') >= 0) p90 = m.forecast_6m[k];
                    }});
                }}
            }});
        }}

        var datasets = [
            {{
                label: 'Historical',
                data: histData,
                borderColor: '{self.KP_BLUE}',
                backgroundColor: 'rgba(0,43,92,0.1)',
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointBackgroundColor: '{self.KP_BLUE}',
                borderWidth: 2.5
            }},
            {{
                label: 'Forecast',
                data: fcData,
                borderColor: '#0066E6',
                borderDash: [6, 3],
                backgroundColor: 'rgba(0,102,230,0.08)',
                tension: 0.3,
                fill: true,
                pointRadius: 5,
                pointBackgroundColor: '#0066E6',
                borderWidth: 2.5
            }}
        ];

        if (p10 && p90) {{
            var bandUpper = new Array(historical.length).fill(null).concat(p90);
            var bandLower = new Array(historical.length).fill(null).concat(p10);
            datasets.push({{
                label: '90th Percentile',
                data: bandUpper,
                borderColor: 'rgba(40,167,69,0.3)',
                backgroundColor: 'rgba(40,167,69,0.08)',
                borderDash: [2, 2],
                fill: false,
                pointRadius: 0,
                borderWidth: 1
            }});
            datasets.push({{
                label: '10th Percentile',
                data: bandLower,
                borderColor: 'rgba(220,53,69,0.3)',
                backgroundColor: 'rgba(220,53,69,0.08)',
                borderDash: [2, 2],
                fill: '-1',
                pointRadius: 0,
                borderWidth: 1
            }});
        }}

        if (_forecastCharts[key]) _forecastCharts[key].destroy();
        _forecastCharts[key] = new Chart(canvas, {{
            type: 'line',
            data: {{ labels: labels, datasets: datasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }} }} }},
                    tooltip: {{
                        callbacks: {{
                            label: function(ctx) {{
                                var v = ctx.parsed.y;
                                return ctx.dataset.label + ': ' + (v !== null ? v.toFixed(2) : 'N/A');
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{ ticks: {{ maxRotation: 45, font: {{ size: 10 }} }} }},
                    y: {{ ticks: {{ callback: function(v) {{ return v.toLocaleString(undefined, {{maximumFractionDigits:1}}); }} }} }}
                }}
            }}
        }});
    }}

    // ═══════════════════════════════════════════════════
    //  AI CHATBOT ENGINE
    // ═══════════════════════════════════════════════════
    var _chatSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2,6);
    var _chatHistory = [];
    var _chatOpen = false;
    var _queryCount = 0;

    function toggleChat() {{
        _chatOpen = !_chatOpen;
        document.getElementById('chatPanel').classList.toggle('open', _chatOpen);
        document.getElementById('chatBadge').style.display = 'none';
        if (_chatOpen) {{
            document.getElementById('chatInput').focus();
            // Update suggestions based on current dashboard context
            updateSuggestions();
        }}
    }}

    function askSuggestion(el) {{
        document.getElementById('chatInput').value = el.textContent;
        sendChat();
    }}

    function sendChat() {{
        var input = document.getElementById('chatInput');
        var q = input.value.trim();
        if (!q) return;
        input.value = '';

        // Add user message
        appendChatMsg('user', q);
        _chatHistory.push({{ role: 'user', content: q, ts: Date.now() }});
        _queryCount++;

        // Show typing indicator
        var typingId = showTyping();

        // Build API URL — try intelligent endpoint first
        var apiBase = window.location.origin || '';
        var url = apiBase + '/api/intelligent/query?q=' + encodeURIComponent(q) + '&session_id=' + _chatSessionId;

        fetch(url)
            .then(function(r) {{ return r.json(); }})
            .then(function(data) {{
                removeTyping(typingId);
                var response = formatChatResponse(data, q);
                appendChatMsg('bot', response);
                _chatHistory.push({{ role: 'bot', content: q, answer: data, ts: Date.now() }});
                // Trigger anticipation after 3+ queries
                if (_queryCount >= 2) triggerAnticipation(q, data);
                // Update suggestions based on what was asked
                updateSuggestionsAfterQuery(q);
            }})
            .catch(function(err) {{
                removeTyping(typingId);
                // Fallback to basic /query endpoint
                fetch(apiBase + '/query?q=' + encodeURIComponent(q))
                    .then(function(r) {{ return r.json(); }})
                    .then(function(data) {{
                        var response = formatChatResponse(data, q);
                        appendChatMsg('bot', response);
                        _chatHistory.push({{ role: 'bot', content: q, answer: data, ts: Date.now() }});
                    }})
                    .catch(function(e2) {{
                        appendChatMsg('bot', 'I encountered an issue processing your request. Please try rephrasing your question. <em style="color:#999;font-size:0.75rem;">(' + (e2.message||'Network error') + ')</em>');
                    }});
            }});
    }}

    function formatChatResponse(data, query) {{
        var html = '';

        // Narrative / explanation
        if (data.narrative) {{
            html += '<div style="margin-bottom:0.5rem;">' + data.narrative + '</div>';
        }} else if (data.explanation) {{
            html += '<div style="margin-bottom:0.5rem;">' + data.explanation + '</div>';
        }}

        // If there are rows, show as table
        if (data.rows && data.rows.length > 0 && data.columns) {{
            html += '<table><thead><tr>';
            data.columns.forEach(function(c) {{ html += '<th>' + c + '</th>'; }});
            html += '</tr></thead><tbody>';
            var maxRows = Math.min(data.rows.length, 8);
            for (var i = 0; i < maxRows; i++) {{
                html += '<tr>';
                data.rows[i].forEach(function(v) {{
                    if (typeof v === 'number') {{
                        html += '<td>' + v.toLocaleString(undefined, {{maximumFractionDigits:2}}) + '</td>';
                    }} else {{
                        html += '<td>' + (v !== null ? v : '—') + '</td>';
                    }}
                }});
                html += '</tr>';
            }}
            html += '</tbody></table>';
            if (data.rows.length > 8) {{
                html += '<div style="font-size:0.72rem;color:#888;margin-top:0.25rem;">Showing 8 of ' + data.rows.length + ' rows</div>';
            }}
        }}

        // Confidence indicator
        if (data.confidence !== undefined) {{
            var confPct = Math.round((data.confidence || 0) * 100);
            var confColor = confPct >= 85 ? '#28a745' : confPct >= 70 ? '#ffc107' : '#dc3545';
            html += '<div style="margin-top:0.5rem;font-size:0.72rem;color:' + confColor + ';">Confidence: ' + confPct + '%</div>';
        }}

        // WHY / HOW / WHAT framework
        if (data.clinical_context || data.business_context) {{
            var ctx = data.clinical_context || data.business_context || {{}};
            if (ctx.why) html += '<div style="margin-top:0.5rem;font-size:0.8rem;"><strong>WHY:</strong> ' + ctx.why + '</div>';
            if (ctx.how) html += '<div style="font-size:0.8rem;"><strong>HOW:</strong> ' + ctx.how + '</div>';
            if (ctx.what) html += '<div style="font-size:0.8rem;"><strong>WHAT:</strong> ' + ctx.what + '</div>';
        }}

        // Recommendations
        if (data.recommendations && data.recommendations.length) {{
            html += '<div style="margin-top:0.5rem;font-size:0.8rem;"><strong>Recommendations:</strong></div>';
            data.recommendations.forEach(function(r) {{
                html += '<div style="font-size:0.78rem;color:#555;padding-left:0.5rem;border-left:2px solid #0052B3;margin:0.25rem 0;">' + (r.recommendation || r) + '</div>';
            }});
        }}

        if (!html) {{
            html = 'I processed your query but didn\'t find specific data matching "<em>' + query + '</em>". Try being more specific — for example, "What is the PMPM revenue by region?" or "Show denial rates for SCAL."';
        }}

        return html;
    }}

    function appendChatMsg(role, html) {{
        var container = document.getElementById('chatMessages');
        var div = document.createElement('div');
        div.className = 'chat-msg ' + role;
        var avatar = role === 'bot' ? 'AI' : 'USER';
        div.innerHTML = '<div class="chat-msg-avatar">' + avatar + '</div><div class="chat-msg-bubble">' + html + '</div>';
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }}

    function showTyping() {{
        var container = document.getElementById('chatMessages');
        var div = document.createElement('div');
        div.className = 'chat-msg bot';
        div.id = 'typing_' + Date.now();
        div.innerHTML = '<div class="chat-msg-avatar">AI</div><div class="chat-msg-bubble"><div class="chat-typing"><span></span><span></span><span></span></div></div>';
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
        return div.id;
    }}

    function removeTyping(id) {{
        var el = document.getElementById(id);
        if (el) el.remove();
    }}

    // ═══════════════════════════════════════════════════
    //  ANTICIPATION ENGINE (Client-side)
    // ═══════════════════════════════════════════════════
    var _intentTransitions = {{
        'mlr': ['pmpm', 'denial_rate', 'expense_breakdown'],
        'pmpm': ['revenue_forecast', 'regional_comparison', 'mlr'],
        'denial_rate': ['denial_by_region', 'recovery_rate', 'coding_analysis'],
        'risk_score': ['high_risk_members', 'risk_distribution', 'intervention'],
        'retention': ['disenrollment_reasons', 'member_satisfaction', 'retention_forecast'],
        'stars': ['hedis_measures', 'cahps_scores', 'quality_improvement'],
        'financial': ['pmpm', 'mlr', 'revenue_forecast'],
        'members': ['enrollment_trend', 'demographics', 'retention']
    }};

    function triggerAnticipation(query, response) {{
        // Determine intent from query keywords
        var qLow = query.toLowerCase();
        var intent = 'general';
        if (qLow.indexOf('mlr') >= 0 || qLow.indexOf('medical loss') >= 0) intent = 'mlr';
        else if (qLow.indexOf('pmpm') >= 0 || qLow.indexOf('revenue') >= 0) intent = 'pmpm';
        else if (qLow.indexOf('denial') >= 0) intent = 'denial_rate';
        else if (qLow.indexOf('risk') >= 0) intent = 'risk_score';
        else if (qLow.indexOf('retention') >= 0 || qLow.indexOf('disenroll') >= 0) intent = 'retention';
        else if (qLow.indexOf('star') >= 0 || qLow.indexOf('hedis') >= 0 || qLow.indexOf('cahps') >= 0) intent = 'stars';
        else if (qLow.indexOf('member') >= 0 || qLow.indexOf('enroll') >= 0) intent = 'members';

        var nextIntents = _intentTransitions[intent];
        if (!nextIntents || !nextIntents.length) return;

        var suggestions = {{
            'pmpm': 'Explore PMPM revenue trends',
            'mlr': 'Review Medical Loss Ratio breakdown',
            'denial_rate': 'Investigate claim denial patterns',
            'denial_by_region': 'Compare denial rates across regions',
            'recovery_rate': 'Check projected denial recovery',
            'coding_analysis': 'Analyze top denial codes',
            'revenue_forecast': 'View 6-month revenue forecast',
            'regional_comparison': 'Compare metrics across KP regions',
            'expense_breakdown': 'Analyze expense categories',
            'risk_distribution': 'See risk score distribution',
            'high_risk_members': 'Identify high-risk member segments',
            'intervention': 'Review intervention recommendations',
            'disenrollment_reasons': 'Understand disenrollment drivers',
            'member_satisfaction': 'Check CAHPS satisfaction scores',
            'retention_forecast': 'Forecast member retention',
            'hedis_measures': 'Deep dive into HEDIS performance',
            'cahps_scores': 'Review patient experience scores',
            'quality_improvement': 'Explore quality improvement areas',
            'enrollment_trend': 'View enrollment trends',
            'demographics': 'Analyze member demographics'
        }};

        var nextSuggestion = suggestions[nextIntents[0]] || 'Explore related metrics';
        var bar = document.getElementById('anticipationBar');
        bar.innerHTML = '<strong>You might want to:</strong> ' + nextSuggestion + ' <span style="cursor:pointer;color:#0052B3;text-decoration:underline;" onclick="askAnticipated(\'' + nextSuggestion + '\')">Ask now →</span>';
        bar.classList.add('show');

        // Auto-hide after 8 seconds
        setTimeout(function() {{ bar.classList.remove('show'); }}, 8000);
    }}

    function askAnticipated(text) {{
        document.getElementById('anticipationBar').classList.remove('show');
        if (!_chatOpen) toggleChat();
        setTimeout(function() {{
            document.getElementById('chatInput').value = text;
            sendChat();
        }}, 300);
    }}

    function updateSuggestions() {{
        // Dynamic suggestions based on dashboard type
        var title = document.title.toLowerCase();
        var container = document.getElementById('chatSuggestions');
        var suggestions = [];

        if (title.indexOf('financial') >= 0) {{
            suggestions = [
                'What is driving our MLR increase?',
                'PMPM breakdown by region',
                'Top denial reasons and recovery potential',
                'Revenue forecast next 6 months',
                'Which region has highest costs?'
            ];
        }} else if (title.indexOf('stars') >= 0 || title.indexOf('quality') >= 0) {{
            suggestions = [
                'Which HEDIS measures are below benchmark?',
                'CAHPS improvement opportunities',
                'What impacts our overall star rating most?',
                'Clinical quality gaps by condition',
                'Admin measure performance trends'
            ];
        }} else if (title.indexOf('executive') >= 0) {{
            suggestions = [
                'Overall performance summary',
                'Which KPIs need immediate attention?',
                'Top 3 strategic priorities',
                'Compare actual vs target across all KPIs',
                'What improved most this quarter?'
            ];
        }} else {{
            suggestions = [
                'What is our current MLR?',
                'Show denial rates by region',
                'Risk score distribution',
                'Revenue forecast next 6 months',
                'Member retention trends'
            ];
        }}

        container.innerHTML = suggestions.map(function(s) {{
            return '<span class="chat-suggestion" onclick="askSuggestion(this)">' + s + '</span>';
        }}).join('');
    }}

    function updateSuggestionsAfterQuery(query) {{
        // After each query, rotate in relevant follow-up suggestions
        var qLow = query.toLowerCase();
        var followUps = [];
        if (qLow.indexOf('mlr') >= 0) followUps = ['MLR trend over 12 months', 'MLR by region', 'What drives MLR up?'];
        else if (qLow.indexOf('denial') >= 0) followUps = ['Top denial codes', 'Denial recovery rate', 'Denials by claim type'];
        else if (qLow.indexOf('risk') >= 0) followUps = ['High risk member count', 'Risk score by region', 'Interventions for high risk'];
        else if (qLow.indexOf('member') >= 0 || qLow.indexOf('enroll') >= 0) followUps = ['Active vs disenrolled', 'Enrollment by region', 'Average member age'];

        if (followUps.length) {{
            var container = document.getElementById('chatSuggestions');
            var existing = Array.from(container.children).map(function(c) {{ return c.textContent; }}).slice(0,2);
            var merged = followUps.concat(existing).slice(0, 5);
            container.innerHTML = merged.map(function(s) {{
                return '<span class="chat-suggestion" onclick="askSuggestion(this)">' + s + '</span>';
            }}).join('');
        }}
    }}

    // ═══════════════════════════════════════════════════
    //  TOOLBAR FUNCTIONS
    // ═══════════════════════════════════════════════════
    function exportDashboard(format) {{
        if (format === 'pdf') {{
            // Hide chatbot elements before printing
            var fab = document.getElementById('chatFab');
            var panel = document.getElementById('chatPanel');
            var toolbar = document.querySelector('.toolbar');
            fab.style.display = 'none';
            panel.style.display = 'none';
            toolbar.style.display = 'none';
            window.print();
            setTimeout(function() {{
                fab.style.display = '';
                panel.style.display = '';
                toolbar.style.display = '';
            }}, 500);
        }} else if (format === 'csv') {{
            // Export all visible tables as CSV
            var tables = document.querySelectorAll('.container table');
            var csv = '';
            tables.forEach(function(table, idx) {{
                if (idx > 0) csv += '\\n\\n';
                var rows = table.querySelectorAll('tr');
                rows.forEach(function(row) {{
                    var cells = row.querySelectorAll('th, td');
                    var rowData = [];
                    cells.forEach(function(cell) {{
                        var text = cell.textContent.replace(/"/g, '""').trim();
                        rowData.push('"' + text + '"');
                    }});
                    csv += rowData.join(',') + '\\n';
                }});
            }});
            if (!csv) {{ alert('No table data found to export.'); return; }}
            var blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            var link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = document.title.replace(/[^a-zA-Z0-9]/g, '_') + '.csv';
            link.click();
        }}
    }}

    function saveDashboardState() {{
        var state = {{
            title: document.title,
            url: window.location.href,
            timestamp: new Date().toISOString(),
            chatHistory: _chatHistory,
            sessionId: _chatSessionId
        }};
        try {{
            var saved = JSON.parse(sessionStorage.getItem('kp_saved_views') || '[]');
            saved.push(state);
            sessionStorage.setItem('kp_saved_views', JSON.stringify(saved));
            showToast('Dashboard view saved successfully');
        }} catch(e) {{
            // Fallback: download as JSON
            var blob = new Blob([JSON.stringify(state, null, 2)], {{ type: 'application/json' }});
            var link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'dashboard_state_' + Date.now() + '.json';
            link.click();
            showToast('Dashboard state downloaded');
        }}
    }}

    var _bookmarked = false;
    function toggleBookmark() {{
        _bookmarked = !_bookmarked;
        var btn = document.getElementById('bookmarkBtn');
        if (_bookmarked) {{
            btn.classList.add('primary');
            showToast('Dashboard bookmarked');
        }} else {{
            btn.classList.remove('primary');
            showToast('Bookmark removed');
        }}
    }}

    function refreshDashboard() {{
        showToast('Refreshing dashboard data...');
        setTimeout(function() {{ window.location.reload(); }}, 500);
    }}

    function showToast(msg) {{
        var toast = document.createElement('div');
        toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#002B5C;color:white;padding:0.75rem 1.25rem;border-radius:8px;font-size:0.85rem;z-index:99999;box-shadow:0 4px 12px rgba(0,0,0,0.2);animation:chatSlideUp 0.3s ease;';
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(function() {{ toast.remove(); }}, 3000);
    }}

    // Set last updated timestamp
    document.addEventListener('DOMContentLoaded', function() {{
        var ts = document.getElementById('lastUpdated');
        if (ts) ts.textContent = 'Updated: ' + new Date().toLocaleString();
    }});

    </script>
    {scripts}
</body>
</html>
"""

    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        if not isinstance(data, dict):
            return default
        return data.get(key, default)

    def _safe_number(self, value: Any, default: str = "0") -> str:
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    return f"{value:,.2f}"
                return f"{value:,}"
            return str(value)
        except:
            return default

    def _build_insight_scripts(self, insights: dict) -> str:
        script = "\n<script>\n"
        for key, data in insights.items():
            try:
                json_str = json.dumps(data, default=str, ensure_ascii=False)
            except:
                json_str = '{}'
            script += f"registerInsight('{key}', {json_str});\n"
        script += "</script>\n"
        return script

    def _rag_status(self, value: float, target: float, tolerance: float = 0.1) -> tuple:
        if value is None or target is None:
            return (self.RAG_YELLOW, "UNKNOWN", "yellow")

        percentage = (value / target * 100) if target > 0 else 0

        if percentage >= (100 - tolerance * 100):
            return (self.RAG_GREEN, "GREEN", "green")
        elif percentage >= (100 - tolerance * 100 * 2):
            return (self.RAG_YELLOW, "YELLOW", "yellow")
        else:
            return (self.RAG_RED, "RED", "red")

    def render_financial_dashboard(self, data: dict) -> str:
        chart_id_expense = self._get_chart_id()
        chart_id_regional = self._get_chart_id()
        chart_id_trend_pmpm = self._get_chart_id()
        chart_id_trend_mlr = self._get_chart_id()

        fin_insights = {}
        if self.insights_engine:
            try:
                fin_insights = self.insights_engine.generate_all_financial_insights()
            except Exception:
                pass

        sections = self._safe_get(data, "sections", {})

        ytd = self._safe_get(sections, "ytd_summary", {})
        ytd_metrics = self._safe_get(ytd, "metrics", {})

        pmpm_revenue_obj = self._safe_get(ytd_metrics, "pmpm_revenue", {})
        pmpm_cost_obj = self._safe_get(ytd_metrics, "pmpm_cost", {})
        mlr_obj = self._safe_get(ytd_metrics, "medical_loss_ratio", {})
        collection_obj = self._safe_get(ytd_metrics, "collection_rate", {})
        total_members_obj = self._safe_get(ytd_metrics, "total_members", {})

        pmpm_revenue = self._safe_get(pmpm_revenue_obj, "value", 0)
        pmpm_cost = self._safe_get(pmpm_cost_obj, "value", 0)
        mlr = self._safe_get(mlr_obj, "value", 0)
        mlr_status = self._safe_get(mlr_obj, "status", "amber")
        mlr_benchmark = self._safe_get(mlr_obj, "benchmark", "")
        collection_rate = self._safe_get(collection_obj, "value", 0)
        collection_status = self._safe_get(collection_obj, "status", "green")
        total_members = self._safe_get(total_members_obj, "value", 0)

        expense_section = self._safe_get(sections, "expense_by_type", {})
        expense_data = self._safe_get(expense_section, "data", [])
        expense_labels = [e.get("claim_type", "") for e in expense_data]
        expense_billed = [round(e.get("billed", 0) / 1_000_000, 2) for e in expense_data]
        expense_paid = [round(e.get("paid", 0) / 1_000_000, 2) for e in expense_data]

        regional_section = self._safe_get(sections, "regional_pl", {})
        regional_data = self._safe_get(regional_section, "data", [])
        region_labels = [r.get("region", "") for r in regional_data]
        region_pmpm = [r.get("pmpm", 0) for r in regional_data]
        region_mlr = [r.get("mlr", 0) for r in regional_data]

        trend_section = self._safe_get(sections, "monthly_trend", {})
        trend_data = self._safe_get(trend_section, "data", [])
        trend_data_sorted = list(reversed(trend_data))
        trend_months = [t.get("month", "") for t in trend_data_sorted]
        trend_pmpm = [t.get("pmpm", 0) for t in trend_data_sorted]
        trend_mlr = [t.get("mlr", 0) for t in trend_data_sorted]

        denial_section = self._safe_get(sections, "denial_savings", {})
        denial_metrics = self._safe_get(denial_section, "metrics", {})
        total_denied = self._safe_get(self._safe_get(denial_metrics, "total_denied", {}), "value", 0)
        projected_recovery = self._safe_get(self._safe_get(denial_metrics, "projected_recovery", {}), "value", 0)
        recovery_rate = self._safe_get(self._safe_get(denial_metrics, "potential_recovery_rate", {}), "value", "60-70%")

        def _status_dot(status):
            return "green" if status == "green" else ("yellow" if status == "amber" else "red")

        content = f"""
        <div class="section-title">YTD Financial Summary</div>
        <div class="grid grid-6">
            <div class="card metric-card clickable" onclick="openInsight('pmpm_revenue')">
                <div class="metric-label">PMPM Revenue (Paid)</div>
                <div class="metric-value">${pmpm_revenue:,.2f}</div>
                <div style="font-size:0.75rem;color:#999;">{self._safe_get(pmpm_revenue_obj, 'benchmark', '')}</div>
            </div>
            <div class="card metric-card clickable" onclick="openInsight('pmpm_revenue')">
                <div class="metric-label">PMPM Cost (Billed)</div>
                <div class="metric-value">${pmpm_cost:,.2f}</div>
            </div>
            <div class="card metric-card clickable" onclick="openInsight('medical_loss_ratio')">
                <div class="metric-label">Medical Loss Ratio</div>
                <div class="metric-value">
                    <span class="status-dot {_status_dot(mlr_status)}" style="margin-right:8px;"></span>{mlr:.2f}%
                </div>
                <div style="font-size:0.75rem;color:#999;">Benchmark: {mlr_benchmark}</div>
            </div>
            <div class="card metric-card clickable" onclick="openInsight('collection_rate')">
                <div class="metric-label">Collection Rate</div>
                <div class="metric-value">
                    <span class="status-dot {_status_dot(collection_status)}" style="margin-right:8px;"></span>{collection_rate:.2f}%
                </div>
                <div style="font-size:0.75rem;color:#999;">{self._safe_get(collection_obj, 'benchmark', '')}</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">Total Members</div>
                <div class="metric-value">{total_members:,}</div>
            </div>
            <div class="card metric-card clickable" onclick="openInsight('denial_rate')">
                <div class="metric-label">Denied Amount</div>
                <div class="metric-value" style="color:{self.RAG_RED};">${total_denied:,.0f}</div>
                <div style="font-size:0.75rem;color:#999;">Recovery: ${projected_recovery:,.0f} ({recovery_rate})</div>
            </div>
        </div>

        <div class="section-title">Expense Breakdown by Claim Type</div>
        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">Billed vs Paid by Claim Type ($M)</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_expense}"></canvas>
                </div>
            </div>
            <div class="card">
                <div class="chart-title">Claim Type Details</div>
                <table>
                    <thead>
                        <tr>
                            <th>Claim Type</th>
                            <th>Claims</th>
                            <th>% of Total</th>
                            <th>Billed</th>
                            <th>Paid</th>
                            <th>MLR</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for e in expense_data:
            ct = e.get("claim_type", "")
            cl = e.get("claims", 0)
            pct = e.get("pct_of_claims", 0)
            billed = e.get("billed", 0)
            paid = e.get("paid", 0)
            eMLR = e.get("mlr", 0)
            content += f"""
                        <tr>
                            <td><strong>{ct}</strong></td>
                            <td>{cl:,}</td>
                            <td>{pct:.1f}%</td>
                            <td>${billed:,.0f}</td>
                            <td>${paid:,.0f}</td>
                            <td>{eMLR:.2f}%</td>
                        </tr>
            """

        content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section-title">Regional Financial Performance</div>
        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">PMPM by Region</div>
                <div class="chart-container">
                    <canvas id="{chart_id_regional}"></canvas>
                </div>
            </div>
            <div class="card">
                <div class="chart-title">Regional P&L Details</div>
                <table>
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Members</th>
                            <th>Claims</th>
                            <th>Billed</th>
                            <th>Paid</th>
                            <th>PMPM</th>
                            <th>MLR</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for r in regional_data:
            rn = r.get("region", "")
            rm = r.get("members", 0)
            rc = r.get("claims", 0)
            rb = r.get("billed", 0)
            rp = r.get("paid", 0)
            rpmpm = r.get("pmpm", 0)
            rmlr = r.get("mlr", 0)
            mlr_class = "green" if rmlr >= 32 else ("yellow" if rmlr >= 30 else "red")
            content += f"""
                        <tr>
                            <td><strong>{rn}</strong></td>
                            <td>{rm:,}</td>
                            <td>{rc:,}</td>
                            <td>${rb:,.0f}</td>
                            <td>${rp:,.0f}</td>
                            <td>${rpmpm:,.2f}</td>
                            <td><span class="status-dot {mlr_class}"></span> {rmlr:.2f}%</td>
                        </tr>
            """

        content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section-title">Monthly PMPM & MLR Trend (Last 12 Months)</div>
        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">PMPM Trend</div>
                <div class="chart-container">
                    <canvas id="{chart_id_trend_pmpm}"></canvas>
                </div>
            </div>
            <div class="card">
                <div class="chart-title">MLR Trend</div>
                <div class="chart-container">
                    <canvas id="{chart_id_trend_mlr}"></canvas>
                </div>
            </div>
        </div>
        """

        scripts = f"""
        <script>
            // Billed vs Paid by Claim Type
            new Chart(document.getElementById('{chart_id_expense}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(expense_labels)},
                    datasets: [
                        {{
                            label: 'Billed ($M)',
                            data: {json.dumps(expense_billed)},
                            backgroundColor: '{self.KP_BLUE}',
                            borderRadius: 4
                        }},
                        {{
                            label: 'Paid ($M)',
                            data: {json.dumps(expense_paid)},
                            backgroundColor: '#4ECDC4',
                            borderRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top', labels: {{ font: {{ size: 12, weight: 'bold' }} }} }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ callback: function(v) {{ return '$' + v.toFixed(0) + 'M'; }} }}
                        }}
                    }}
                }}
            }});

            // Regional PMPM + MLR
            new Chart(document.getElementById('{chart_id_regional}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(region_labels)},
                    datasets: [
                        {{
                            label: 'PMPM ($)',
                            data: {json.dumps(region_pmpm)},
                            backgroundColor: '{self.KP_BLUE}',
                            borderRadius: 4,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'MLR (%)',
                            data: {json.dumps(region_mlr)},
                            type: 'line',
                            borderColor: '#FF6B6B',
                            backgroundColor: 'rgba(255,107,107,0.1)',
                            tension: 0.3,
                            pointRadius: 5,
                            pointBackgroundColor: '#FF6B6B',
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top', labels: {{ font: {{ size: 12, weight: 'bold' }} }} }}
                    }},
                    scales: {{
                        y: {{
                            type: 'linear',
                            position: 'left',
                            beginAtZero: true,
                            ticks: {{ callback: function(v) {{ return '$' + v.toFixed(0); }} }}
                        }},
                        y1: {{
                            type: 'linear',
                            position: 'right',
                            min: 25,
                            max: 40,
                            ticks: {{ callback: function(v) {{ return v.toFixed(0) + '%'; }} }},
                            grid: {{ drawOnChartArea: false }}
                        }}
                    }}
                }}
            }});

            // Monthly PMPM Trend
            new Chart(document.getElementById('{chart_id_trend_pmpm}'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(trend_months)},
                    datasets: [{{
                        label: 'PMPM ($)',
                        data: {json.dumps(trend_pmpm)},
                        borderColor: '{self.KP_BLUE}',
                        backgroundColor: 'rgba(0, 43, 92, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 5,
                        pointBackgroundColor: '{self.KP_BLUE}',
                        pointBorderColor: '{self.KP_WHITE}',
                        pointBorderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ labels: {{ font: {{ size: 12, weight: 'bold' }} }} }} }},
                    scales: {{
                        y: {{
                            ticks: {{ callback: function(v) {{ return '$' + v.toFixed(0); }} }}
                        }}
                    }}
                }}
            }});

            // Monthly MLR Trend
            new Chart(document.getElementById('{chart_id_trend_mlr}'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(trend_months)},
                    datasets: [{{
                        label: 'MLR (%)',
                        data: {json.dumps(trend_mlr)},
                        borderColor: '#FF6B6B',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 5,
                        pointBackgroundColor: '#FF6B6B',
                        pointBorderColor: '{self.KP_WHITE}',
                        pointBorderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ labels: {{ font: {{ size: 12, weight: 'bold' }} }} }} }},
                    scales: {{
                        y: {{
                            ticks: {{ callback: function(v) {{ return v.toFixed(0) + '%'; }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

        insight_scripts = self._build_insight_scripts(fin_insights) if fin_insights else ""
        scripts += insight_scripts

        return self._get_base_html("KP Medicare Advantage — Financial Performance Dashboard", content, scripts)

    def render_stars_dashboard(self, data: dict) -> str:
        chart_id_hedis = self._get_chart_id()
        chart_id_cahps = self._get_chart_id()
        chart_id_radar = self._get_chart_id()

        stars_insights = {}
        if self.insights_engine:
            try:
                stars_insights = self.insights_engine.generate_all_stars_insights(data)
            except Exception:
                pass

        sections = self._safe_get(data, "sections", {})

        overall_section = self._safe_get(sections, "overall_rating", {})
        overall_rating = self._safe_get(overall_section, "rating", 0)
        max_rating = self._safe_get(overall_section, "max_rating", 5.0)
        overall_benchmark = self._safe_get(overall_section, "benchmark", 4.0)
        overall_status = self._safe_get(overall_section, "status", "amber")
        composition = self._safe_get(overall_section, "composition", {})

        full_stars = int(overall_rating)
        half_star = (overall_rating % 1) >= 0.25
        star_html = '<span style="color:#ffc107;font-size:2.5rem;">' + ("&#9733;" * full_stars)
        if half_star and full_stars < 5:
            star_html += "&#9734;"
            remaining = 5 - full_stars - 1
        else:
            remaining = 5 - full_stars
        star_html += '<span style="color:#ddd;">' + ("&#9734;" * max(remaining, 0)) + "</span></span>"

        def _status_dot(status):
            return "green" if status == "green" else ("yellow" if status == "amber" else "red")

        def _status_color(status):
            if status == "green": return self.RAG_GREEN
            if status == "amber": return self.RAG_YELLOW
            return self.RAG_RED

        hedis_section = self._safe_get(sections, "hedis_measures", {})
        hedis_measures = self._safe_get(hedis_section, "measures", [])
        hedis_labels = [m.get("measure", "")[:30] for m in hedis_measures]
        hedis_rates = [m.get("rate", 0) for m in hedis_measures]
        hedis_benchmarks = [m.get("benchmark", 0) for m in hedis_measures]

        cahps_section = self._safe_get(sections, "cahps_measures", {})
        cahps_measures = self._safe_get(cahps_section, "measures", [])

        clinical_section = self._safe_get(sections, "clinical_quality", {})
        clinical_data = self._safe_get(clinical_section, "data", [])

        admin_section = self._safe_get(sections, "admin_measures", {})
        admin_measures = self._safe_get(admin_section, "measures", [])

        def _avg_pct(measures_list):
            vals = []
            for m in measures_list:
                rate = m.get("rate", m.get("value", 0))
                bench = m.get("benchmark", m.get("5_star_cut", 100))
                if bench and bench > 0:
                    vals.append(min((rate / bench) * 100, 120))
            return round(sum(vals) / len(vals), 1) if vals else 0

        hedis_pct = _avg_pct(hedis_measures)
        cahps_pct = _avg_pct(cahps_measures)
        clinical_pct = 50.0
        admin_pct = _avg_pct(admin_measures)

        content = f"""
        <div class="card card-full" style="text-align:center;padding:2rem;">
            <h2 style="color:{self.KP_BLUE};margin-bottom:0.5rem;">Overall CMS Star Rating</h2>
            <div style="margin:1rem 0;">{star_html}</div>
            <div style="font-size:1.75rem;font-weight:700;color:{self.KP_BLUE};">{overall_rating:.2f} / {max_rating:.1f}</div>
            <div style="margin-top:0.5rem;">
                <span class="status-badge {_status_dot(overall_status)}" style="font-size:0.9rem;">
                    <span class="status-dot {_status_dot(overall_status)}"></span>
                    {overall_status.upper()} — Benchmark: {overall_benchmark} Stars
                </span>
            </div>
            <div style="margin-top:1rem;font-size:0.8rem;color:#888;">
                Weights: HEDIS {composition.get('hedis_weight', 0)*100:.0f}% |
                CAHPS {composition.get('cahps_weight', 0)*100:.0f}% |
                Clinical {composition.get('clinical_quality_weight', 0)*100:.0f}% |
                Admin {composition.get('admin_weight', 0)*100:.0f}% |
                Other {composition.get('other_weight', 0)*100:.0f}%
            </div>
        </div>

        <div class="section-title">HEDIS Preventive Care Measures</div>
        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">HEDIS Rates vs Benchmarks</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_hedis}"></canvas>
                </div>
            </div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Measure</th>
                            <th>Num / Denom</th>
                            <th>Rate</th>
                            <th>5-Star Cut</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for idx, m in enumerate(hedis_measures):
            name = m.get("measure", "")
            num = m.get("numerator", "")
            denom = m.get("denominator", "")
            rate = m.get("rate", 0)
            bench = m.get("benchmark", 0)
            status = m.get("status", "amber")
            note = m.get("note", "")
            frac = f"{num:,} / {denom:,}" if isinstance(num, int) else f"{num} / {denom}"
            insight_key = f"hedis_{idx}"
            content += f"""
                        <tr class="clickable" style="cursor:pointer;" onclick="openInsight('{insight_key}')">
                            <td><strong>{name}</strong><br><small style="color:#888;">{note}</small></td>
                            <td>{frac}</td>
                            <td style="font-weight:700;color:{_status_color(status)};">{rate:.1f}%</td>
                            <td>{bench}%</td>
                            <td><span class="status-badge {_status_dot(status)}"><span class="status-dot {_status_dot(status)}"></span> {status.upper()}</span></td>
                        </tr>
            """

        content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section-title">CAHPS Member Satisfaction</div>
        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">CAHPS Scores vs 5-Star Cuts</div>
                <div class="chart-container">
                    <canvas id="{chart_id_cahps}"></canvas>
                </div>
            </div>
            <div class="card">
                <table>
                    <thead>
                        <tr>
                            <th>Measure</th>
                            <th>Rate</th>
                            <th>5-Star Cut</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        cahps_labels = []
        cahps_rates = []
        cahps_benchmarks_vals = []
        for idx_c, m in enumerate(cahps_measures):
            name = m.get("measure", "")
            rate = m.get("rate", 0)
            bench = m.get("benchmark", m.get("5_star_cut", 0))
            status = m.get("status", "amber")
            cahps_labels.append(name[:25])
            cahps_rates.append(rate)
            cahps_benchmarks_vals.append(bench)
            insight_key_c = f"cahps_{idx_c}"
            content += f"""
                        <tr class="clickable" style="cursor:pointer;" onclick="openInsight('{insight_key_c}')">
                            <td><strong>{name}</strong></td>
                            <td style="font-weight:700;color:{_status_color(status)};">{rate:.1f}%</td>
                            <td>{bench}%</td>
                            <td><span class="status-badge {_status_dot(status)}"><span class="status-dot {_status_dot(status)}"></span> {status.upper()}</span></td>
                        </tr>
            """

        content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section-title">Clinical Quality Measures</div>
        <div class="card card-full">
            <table>
                <thead>
                    <tr>
                        <th>Condition</th>
                        <th>Patients</th>
                        <th>Control Rate</th>
                        <th>Avg Visit Frequency</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """

        for idx_cl, c in enumerate(clinical_data):
            cond = c.get("condition", "")
            patients = c.get("patients", 0)
            cr = c.get("control_rate", 0)
            vf = c.get("visit_frequency", 0)
            cr_status = "green" if cr > 0 else ("yellow" if cr > -10 else "red")
            insight_key_cl = f"clinical_{idx_cl}"
            content += f"""
                    <tr class="clickable" style="cursor:pointer;" onclick="openInsight('{insight_key_cl}')">
                        <td><strong>{cond}</strong></td>
                        <td>{patients:,}</td>
                        <td style="font-weight:700;color:{_status_color(cr_status)};">{cr:+.2f}%</td>
                        <td>{vf:.1f}</td>
                        <td><span class="status-dot {_status_dot(cr_status)}"></span> {'Improving' if cr > 0 else 'Declining'}</td>
                    </tr>
            """

        content += f"""
                </tbody>
            </table>
        </div>

        <div class="section-title">Administrative Performance</div>
        <div class="grid grid-3">
        """

        for idx_a, m in enumerate(admin_measures):
            name = m.get("measure", "")
            rate = m.get("rate", m.get("value", 0))
            bench = m.get("benchmark", m.get("5_star_cut", 0))
            status = m.get("status", "amber")
            is_lower_better = "time" in name.lower() or "pending" in name.lower()
            if is_lower_better:
                pct_of_target = min((bench / rate * 100) if rate > 0 else 0, 100)
            else:
                pct_of_target = min((rate / bench * 100) if bench > 0 else 0, 100)
            insight_key_a = f"admin_{idx_a}"
            content += f"""
            <div class="card clickable" onclick="openInsight('{insight_key_a}')">
                <div class="chart-title">{name}</div>
                <div style="text-align:center;padding:1rem;">
                    <div style="font-size:2.5rem;font-weight:700;color:{_status_color(status)};">{rate:.1f}{'%' if 'rate' in m or '%' in name.lower() else ''}</div>
                    <div style="font-size:0.8rem;color:#888;margin:0.5rem 0;">5-Star Cut: {bench}</div>
                    <span class="status-badge {_status_dot(status)}"><span class="status-dot {_status_dot(status)}"></span> {status.upper()}</span>
                    <div class="progress-bar" style="margin-top:1rem;">
                        <div class="progress-fill" style="width:{pct_of_target:.0f}%;background:{'linear-gradient(90deg,' + _status_color(status) + ',' + _status_color(status) + ')'}"></div>
                    </div>
                </div>
            </div>
            """

        content += f"""
        </div>

        <div class="section-title">Performance Radar</div>
        <div class="card card-full">
            <div class="chart-container-large">
                <canvas id="{chart_id_radar}"></canvas>
            </div>
        </div>
        """

        scripts = f"""
        <script>
            // HEDIS Bar Chart
            new Chart(document.getElementById('{chart_id_hedis}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(hedis_labels)},
                    datasets: [
                        {{
                            label: 'Actual Rate (%)',
                            data: {json.dumps(hedis_rates)},
                            backgroundColor: '{self.KP_BLUE}',
                            borderRadius: 4
                        }},
                        {{
                            label: '5-Star Benchmark (%)',
                            data: {json.dumps(hedis_benchmarks)},
                            backgroundColor: 'rgba(40,167,69,0.3)',
                            borderColor: '{self.RAG_GREEN}',
                            borderWidth: 2,
                            borderRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
                    scales: {{
                        x: {{ beginAtZero: true, ticks: {{ callback: function(v) {{ return v + '%'; }} }} }}
                    }}
                }}
            }});

            // CAHPS Bar Chart
            new Chart(document.getElementById('{chart_id_cahps}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(cahps_labels)},
                    datasets: [
                        {{
                            label: 'Actual Rate (%)',
                            data: {json.dumps(cahps_rates)},
                            backgroundColor: '{self.KP_BLUE}',
                            borderRadius: 4
                        }},
                        {{
                            label: '5-Star Cut (%)',
                            data: {json.dumps(cahps_benchmarks_vals)},
                            backgroundColor: 'rgba(40,167,69,0.3)',
                            borderColor: '{self.RAG_GREEN}',
                            borderWidth: 2,
                            borderRadius: 4
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }} }},
                    scales: {{
                        x: {{ beginAtZero: true, max: 100, ticks: {{ callback: function(v) {{ return v + '%'; }} }} }}
                    }}
                }}
            }});

            // Radar Chart
            new Chart(document.getElementById('{chart_id_radar}'), {{
                type: 'radar',
                data: {{
                    labels: ['HEDIS', 'CAHPS', 'Clinical Quality', 'Administrative'],
                    datasets: [{{
                        label: 'Performance vs Benchmark (%)',
                        data: [{hedis_pct}, {cahps_pct}, {clinical_pct}, {admin_pct}],
                        borderColor: '{self.KP_BLUE}',
                        backgroundColor: 'rgba(0, 43, 92, 0.15)',
                        pointBackgroundColor: '{self.KP_BLUE}',
                        pointBorderColor: '{self.KP_WHITE}',
                        pointBorderWidth: 2,
                        pointRadius: 6
                    }},
                    {{
                        label: 'Target (100%)',
                        data: [100, 100, 100, 100],
                        borderColor: '{self.RAG_GREEN}',
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        pointRadius: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 12 }} }} }} }},
                    scales: {{
                        r: {{ min: 0, max: 120, beginAtZero: true, ticks: {{ callback: function(v) {{ return v + '%'; }} }} }}
                    }}
                }}
            }});
        </script>
        """

        insight_scripts = self._build_insight_scripts(stars_insights) if stars_insights else ""
        scripts += insight_scripts

        return self._get_base_html("KP Medicare Advantage — Stars Quality Dashboard", content, scripts)

    def render_member_experience(self, data: dict) -> str:
        chart_id_grievance = self._get_chart_id()

        satisfaction_score = self._safe_get(data, "satisfaction_score", 75)
        retention_rate = self._safe_get(data, "retention_rate", 85)

        grievances = self._safe_get(data, "grievance_trend", {})
        grievance_months = list(grievances.keys()) if grievances else []
        grievance_values = list(grievances.values()) if grievances else []

        complaints = self._safe_get(data, "top_complaints", {})
        complaint_categories = list(complaints.keys()) if complaints else []
        complaint_counts = list(complaints.values()) if complaints else []

        regional_satisfaction = self._safe_get(data, "regional_satisfaction", [])

        content = f"""
        <div class="grid grid-3">
            <div class="card">
                <div class="chart-title">Satisfaction Score</div>
                <div style="text-align: center; padding: 1rem 0;">
                    <div style="font-size: 3rem; font-weight: 700; color: {self.KP_BLUE};">{satisfaction_score}%</div>
                    <div style="font-size: 0.875rem; color: #666; margin-top: 0.5rem;">Overall Satisfaction</div>
                    <div class="progress-bar" style="margin-top: 1rem;">
                        <div class="progress-fill" style="width: {satisfaction_score}%"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">Retention Rate</div>
                <div style="text-align: center; padding: 1rem 0;">
                    <div style="font-size: 3rem; font-weight: 700; color: {self.KP_BLUE};">{retention_rate}%</div>
                    <div style="font-size: 0.875rem; color: #666; margin-top: 0.5rem;">Member Retention</div>
                    <div class="progress-bar" style="margin-top: 1rem;">
                        <div class="progress-fill" style="width: {retention_rate}%"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">YoY Change</div>
                <div style="text-align: center; padding: 1rem 0;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: {self.RAG_GREEN};">+3.2%</div>
                    <div style="font-size: 0.875rem; color: #666; margin-top: 0.5rem;">Member Growth</div>
                </div>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card card-full">
                <div class="chart-title">Grievance Trend</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_grievance}"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">Top Complaint Categories</div>
                <table>
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Count</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for category, count in zip(complaint_categories, complaint_counts):
            content += f"""
                        <tr>
                            <td>{category}</td>
                            <td><strong>{count}</strong></td>
                        </tr>
            """

        content += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card card-full">
            <div class="chart-title">Regional Satisfaction Heatmap</div>
            <table class="heatmap-table">
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>Access to Care</th>
                        <th>Care Quality</th>
                        <th>Communication</th>
                        <th>Overall</th>
                    </tr>
                </thead>
                <tbody>
        """

        for region in regional_satisfaction:
            region_name = self._safe_get(region, "region", "Unknown")
            access = self._safe_get(region, "access_to_care", 75)
            quality = self._safe_get(region, "care_quality", 75)
            communication = self._safe_get(region, "communication", 75)
            overall = self._safe_get(region, "overall", 75)

            def heatmap_class(score):
                if score >= 85:
                    return "high"
                elif score >= 70:
                    return "medium"
                else:
                    return "low"

            content += f"""
                    <tr>
                        <td><strong>{region_name}</strong></td>
                        <td><div class="heatmap-cell {heatmap_class(access)}">{access}%</div></td>
                        <td><div class="heatmap-cell {heatmap_class(quality)}">{quality}%</div></td>
                        <td><div class="heatmap-cell {heatmap_class(communication)}">{communication}%</div></td>
                        <td><div class="heatmap-cell {heatmap_class(overall)}">{overall}%</div></td>
                    </tr>
            """

        content += """
                </tbody>
            </table>
        </div>
        """

        scripts = f"""
        <script>
            new Chart(document.getElementById('{chart_id_grievance}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(grievance_months)},
                    datasets: [{{
                        label: 'Grievances',
                        data: {json.dumps(grievance_values)},
                        backgroundColor: '#FF6B6B',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        </script>
        """

        return self._get_base_html("Member Experience Dashboard", content, scripts)

    def render_rada_dashboard(self, data: dict) -> str:
        chart_id_risk = self._get_chart_id()
        chart_id_impact = self._get_chart_id()

        risk_scores = self._safe_get(data, "risk_score_distribution", {})
        hcc_capture_rate = self._safe_get(data, "hcc_capture_rate", 75)
        revenue_impact = self._safe_get(data, "revenue_impact", {})
        coding_accuracy = self._safe_get(data, "coding_accuracy_metrics", {})
        regional_risk = self._safe_get(data, "regional_risk_scores", [])

        risk_bins = list(risk_scores.keys()) if risk_scores else []
        risk_counts = list(risk_scores.values()) if risk_scores else []

        impact_categories = list(revenue_impact.keys()) if revenue_impact else []
        impact_values = list(revenue_impact.values()) if revenue_impact else []

        content = f"""
        <div class="grid grid-2">
            <div class="card metric-card">
                <div class="metric-label">HCC Capture Rate</div>
                <div class="metric-value">{hcc_capture_rate:.1f}%</div>
                <div class="metric-change positive">↑ 2.5%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">Avg Risk Score</div>
                <div class="metric-value">1.24</div>
                <div class="metric-change positive">↑ 0.08</div>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card card-full">
                <div class="chart-title">Risk Score Distribution</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_risk}"></canvas>
                </div>
            </div>

            <div class="card card-full">
                <div class="chart-title">Revenue Impact by Category</div>
                <div class="chart-container">
                    <canvas id="{chart_id_impact}"></canvas>
                </div>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card card-full">
                <div class="chart-title">Coding Accuracy Metrics</div>
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Current</th>
                            <th>Target</th>
                            <th>Variance</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for metric_name, values in coding_accuracy.items():
            current = self._safe_get(values, "current", 0)
            target = self._safe_get(values, "target", 100)
            variance = current - target if isinstance(current, (int, float)) and isinstance(target, (int, float)) else 0

            color, label, dot_class = self._rag_status(current, target)

            content += f"""
                        <tr>
                            <td><strong>{metric_name}</strong></td>
                            <td>{current}%</td>
                            <td>{target}%</td>
                            <td><span class="status-dot {dot_class}"></span> {variance:+.1f}%</td>
                        </tr>
            """

        content += """
                    </tbody>
                </table>
            </div>

            <div class="card card-full">
                <div class="chart-title">Regional Risk Score Comparison</div>
                <table>
                    <thead>
                        <tr>
                            <th>Region</th>
                            <th>Avg Risk Score</th>
                            <th>HCC Count</th>
                            <th>Capture Rate</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for region in regional_risk:
            region_name = self._safe_get(region, "region", "Unknown")
            avg_risk = self._safe_get(region, "avg_risk_score", 0)
            hcc_count = self._safe_get(region, "hcc_count", 0)
            capture = self._safe_get(region, "capture_rate", 0)

            color, label, dot_class = self._rag_status(capture, 85)

            content += f"""
                        <tr>
                            <td><strong>{region_name}</strong></td>
                            <td>{avg_risk:.2f}</td>
                            <td>{hcc_count}</td>
                            <td>{capture:.1f}%</td>
                            <td><span class="status-dot {dot_class}"></span> {label}</td>
                        </tr>
            """

        content += """
                    </tbody>
                </table>
            </div>
        </div>
        """

        scripts = f"""
        <script>
            new Chart(document.getElementById('{chart_id_risk}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(risk_bins)},
                    datasets: [{{
                        label: 'Number of Members',
                        data: {json.dumps(risk_counts)},
                        backgroundColor: '{self.KP_BLUE}',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});

            new Chart(document.getElementById('{chart_id_impact}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(impact_categories)},
                    datasets: [{{
                        label: 'Revenue Impact ($)',
                        data: {json.dumps(impact_values)},
                        backgroundColor: ['#28a745', '#ffc107', '#FF6B6B'],
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ callback: function(value) {{ return '$' + (value / 1000).toFixed(0) + 'K'; }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

        return self._get_base_html("RADA Risk Adjustment Dashboard", content, scripts)

    def render_membership_dashboard(self, data: dict) -> str:
        chart_id_trend = self._get_chart_id()
        chart_id_market = self._get_chart_id()
        chart_id_plans = self._get_chart_id()

        total_members = self._safe_get(data, "total_members", 0)
        growth_rate = self._safe_get(data, "growth_rate", 0)
        enrollment_trend = self._safe_get(data, "enrollment_trend", {})
        market_share = self._safe_get(data, "market_share_by_region", {})
        plan_distribution = self._safe_get(data, "plan_type_distribution", {})

        trend_months = list(enrollment_trend.keys()) if enrollment_trend else []
        enrollment_values = list(enrollment_trend.values()) if enrollment_trend else []

        market_regions = list(market_share.keys()) if market_share else []
        market_values = list(market_share.values()) if market_share else []

        plan_types = list(plan_distribution.keys()) if plan_distribution else []
        plan_values = list(plan_distribution.values()) if plan_distribution else []

        content = f"""
        <div class="grid grid-3">
            <div class="card metric-card">
                <div class="metric-label">Total Members</div>
                <div class="metric-value">{self._safe_number(total_members)}</div>
                <div class="metric-change positive">↑ {growth_rate}%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">YoY Growth</div>
                <div class="metric-value">{growth_rate}%</div>
                <div class="metric-change positive">↑ 0.5%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">Net Additions</div>
                <div class="metric-value">12,450</div>
                <div class="metric-change positive">↑ 8.2%</div>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card card-full">
                <div class="chart-title">Enrollment Trend</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_trend}"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">Market Share by Region</div>
                <div class="chart-container">
                    <canvas id="{chart_id_market}"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">Plan Type Distribution</div>
                <div class="chart-container">
                    <canvas id="{chart_id_plans}"></canvas>
                </div>
            </div>
        </div>
        """

        scripts = f"""
        <script>
            new Chart(document.getElementById('{chart_id_trend}'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(trend_months)},
                    datasets: [{{
                        label: 'Member Enrollment',
                        data: {json.dumps(enrollment_values)},
                        borderColor: '{self.KP_BLUE}',
                        backgroundColor: 'rgba(0, 43, 92, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 5,
                        pointBackgroundColor: '{self.KP_BLUE}',
                        pointBorderColor: '{self.KP_WHITE}',
                        pointBorderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ callback: function(value) {{ return (value / 1000).toFixed(0) + 'K'; }} }}
                        }}
                    }}
                }}
            }});

            new Chart(document.getElementById('{chart_id_market}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(market_regions)},
                    datasets: [{{
                        label: 'Market Share %',
                        data: {json.dumps(market_values)},
                        backgroundColor: '{self.KP_BLUE}',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{ callback: function(value) {{ return value + '%'; }} }}
                        }}
                    }}
                }}
            }});

            new Chart(document.getElementById('{chart_id_plans}'), {{
                type: 'doughnut',
                data: {{
                    labels: {json.dumps(plan_types)},
                    datasets: [{{
                        data: {json.dumps(plan_values)},
                        backgroundColor: ['{self.KP_BLUE}', '#4ECDC4', '#FF6B6B', '#FFD93D'],
                        borderColor: '{self.KP_WHITE}',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{ font: {{ size: 11 }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

        return self._get_base_html("Membership & Enrollment Dashboard", content, scripts)

    def render_utilization_dashboard(self, data: dict) -> str:
        chart_id_util = self._get_chart_id()
        chart_id_cost = self._get_chart_id()

        ip_per_1000 = self._safe_get(data, "ip_per_1000", 0)
        ed_per_1000 = self._safe_get(data, "ed_per_1000", 0)
        op_per_1000 = self._safe_get(data, "op_per_1000", 0)

        utilization_by_type = self._safe_get(data, "utilization_by_type", {})
        unit_costs = self._safe_get(data, "unit_cost_comparison", {})
        regional_utilization = self._safe_get(data, "regional_utilization", [])

        util_types = list(utilization_by_type.keys()) if utilization_by_type else []
        util_values = list(utilization_by_type.values()) if utilization_by_type else []

        cost_types = list(unit_costs.keys()) if unit_costs else []
        cost_values = list(unit_costs.values()) if unit_costs else []

        content = f"""
        <div class="grid grid-3">
            <div class="card metric-card">
                <div class="metric-label">IP per 1000</div>
                <div class="metric-value">{ip_per_1000:.1f}</div>
                <div class="metric-change positive">↓ 2.3%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">ED per 1000</div>
                <div class="metric-value">{ed_per_1000:.1f}</div>
                <div class="metric-change negative">↑ 1.1%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">OP per 1000</div>
                <div class="metric-value">{op_per_1000:.1f}</div>
                <div class="metric-change positive">↓ 3.5%</div>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card card-full">
                <div class="chart-title">Utilization by Visit Type</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_util}"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">Unit Cost Comparison</div>
                <div class="chart-container">
                    <canvas id="{chart_id_cost}"></canvas>
                </div>
            </div>
        </div>

        <div class="card card-full">
            <div class="chart-title">Regional Utilization Metrics</div>
            <table>
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>IP/1000</th>
                        <th>ED/1000</th>
                        <th>OP/1000</th>
                        <th>Overall Status</th>
                    </tr>
                </thead>
                <tbody>
        """

        for region in regional_utilization:
            region_name = self._safe_get(region, "region", "Unknown")
            ip = self._safe_get(region, "ip_per_1000", 0)
            ed = self._safe_get(region, "ed_per_1000", 0)
            op = self._safe_get(region, "op_per_1000", 0)
            status = self._safe_get(region, "status", "GREEN")

            dot_class = "green" if status == "GREEN" else ("yellow" if status == "YELLOW" else "red")

            content += f"""
                    <tr>
                        <td><strong>{region_name}</strong></td>
                        <td>{ip:.1f}</td>
                        <td>{ed:.1f}</td>
                        <td>{op:.1f}</td>
                        <td><span class="status-dot {dot_class}"></span> {status}</td>
                    </tr>
            """

        content += """
                </tbody>
            </table>
        </div>
        """

        scripts = f"""
        <script>
            new Chart(document.getElementById('{chart_id_util}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(util_types)},
                    datasets: [{{
                        label: 'Utilization Rate (per 1000)',
                        data: {json.dumps(util_values)},
                        backgroundColor: '{self.KP_BLUE}',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});

            new Chart(document.getElementById('{chart_id_cost}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(cost_types)},
                    datasets: [{{
                        label: 'Unit Cost ($)',
                        data: {json.dumps(cost_values)},
                        backgroundColor: '#FF6B6B',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ callback: function(value) {{ return '$' + value.toLocaleString(); }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

        return self._get_base_html("Utilization Dashboard", content, scripts)

    def render_executive_summary(self, data: dict) -> str:
        chart_id_kpi = self._get_chart_id()

        sections = self._safe_get(data, "sections", {})

        exec_insights = {}
        if self.insights_engine:
            try:
                exec_insights = self.insights_engine.generate_all_executive_insights()
            except Exception:
                pass

        scorecard = self._safe_get(sections, "performance_scorecard", {})
        overall_status = self._safe_get(scorecard, "overall_status", "amber")
        rag_summary = self._safe_get(scorecard, "rag_summary", {})
        kpis = self._safe_get(scorecard, "kpis", [])

        green_count = self._safe_get(rag_summary, "green", 0)
        amber_count = self._safe_get(rag_summary, "amber", 0)
        red_count = self._safe_get(rag_summary, "red", 0)

        priorities_section = self._safe_get(sections, "strategic_priorities", {})
        priorities = self._safe_get(priorities_section, "priorities", [])

        highlights_section = self._safe_get(sections, "highlights", {})
        positive_areas = self._safe_get(highlights_section, "positive_areas", [])

        concerns_section = self._safe_get(sections, "concerns", {})
        concerns = self._safe_get(concerns_section, "concerns", [])

        def _status_dot(status):
            return "green" if status == "green" else ("yellow" if status == "amber" else "red")

        def _status_color(status):
            if status == "green": return self.RAG_GREEN
            if status == "amber": return self.RAG_YELLOW
            return self.RAG_RED

        def _overall_label(status):
            if status == "green": return "ON TRACK"
            if status == "amber": return "NEEDS ATTENTION"
            return "ACTION REQUIRED"

        kpi_labels = [k.get("metric", "")[:25] for k in kpis]
        kpi_values = [k.get("value", 0) for k in kpis]
        kpi_targets = [k.get("target", 0) for k in kpis]
        kpi_colors = [_status_color(k.get("status", "amber")) for k in kpis]

        content = f"""
        <div class="card card-full" style="text-align:center;padding:2rem;border-left:6px solid {_status_color(overall_status)};">
            <div style="display:flex;align-items:center;justify-content:center;gap:2rem;flex-wrap:wrap;">
                <div>
                    <h2 style="color:{self.KP_BLUE};margin-bottom:0.5rem;">KP Performance Scorecard</h2>
                    <div style="font-size:1.25rem;font-weight:700;color:{_status_color(overall_status)};">
                        {_overall_label(overall_status)}
                    </div>
                </div>
                <div style="display:flex;gap:1.5rem;margin-top:0.5rem;">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;font-weight:700;color:{self.RAG_GREEN};">{green_count}</div>
                        <div style="font-size:0.75rem;color:#666;">GREEN</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem;font-weight:700;color:{self.RAG_YELLOW};">{amber_count}</div>
                        <div style="font-size:0.75rem;color:#666;">AMBER</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem;font-weight:700;color:{self.RAG_RED};">{red_count}</div>
                        <div style="font-size:0.75rem;color:#666;">RED</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="section-title">Key Performance Indicators</div>
        <div class="grid grid-6">
        """

        _metric_insight_map = {
            'Member Retention Rate': 'member_retention_rate',
            'Medical Loss Ratio': 'medical_loss_ratio',
            'Claims Collection Rate': 'claims_collection_rate',
            'Denial Rate': 'denial_rate',
            'Average Risk Score': 'average_risk_score'
        }

        for kpi in kpis:
            metric = kpi.get("metric", "")
            value = kpi.get("value", 0)
            unit = kpi.get("unit", "")
            target = kpi.get("target", 0)
            benchmark = kpi.get("benchmark", "")
            status = kpi.get("status", "amber")

            if unit == "%":
                val_display = f"{value:.2f}%"
                tgt_display = f"{target}%"
            elif unit == "score":
                val_display = f"{value:.3f}"
                tgt_display = f"{target}"
            else:
                val_display = f"{value:,.2f}"
                tgt_display = f"{target}"

            insight_key = _metric_insight_map.get(metric, metric.lower().replace(' ', '_'))
            onclick = f'onclick="openInsight(\'{insight_key}\')"'

            content += f"""
            <div class="card metric-card clickable" style="border-top:4px solid {_status_color(status)};" {onclick}>
                <div class="metric-label">{metric}</div>
                <div class="metric-value" style="font-size:1.75rem;">{val_display}</div>
                <div style="font-size:0.75rem;color:#999;margin-top:0.25rem;">Target: {tgt_display}</div>
                <div style="font-size:0.7rem;color:#aaa;">{benchmark}</div>
                <span class="status-badge {_status_dot(status)}" style="margin-top:0.5rem;">
                    <span class="status-dot {_status_dot(status)}"></span> {status.upper()}
                </span>
            </div>
            """

        content += f"""
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">KPI Values vs Targets</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_kpi}"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="section-title" style="margin-top:0;">Performance Highlights</div>
        """

        if positive_areas:
            for h in positive_areas:
                hm = h.get("metric", "")
                hv = h.get("value", 0)
                hu = h.get("unit", "")
                hb = h.get("benchmark", "")
                content += f"""
                <div class="action-item" style="border-left-color:{self.RAG_GREEN};">
                    <div class="action-item-title" style="color:{self.RAG_GREEN};">{hm}</div>
                    <div class="action-item-desc">{hv}{hu} — Benchmark: {hb}</div>
                </div>
                """

        content += f"""
                <div class="section-title" style="color:{self.RAG_RED};">Areas Requiring Attention</div>
        """

        if concerns:
            for c in concerns:
                cm = c.get("metric", "")
                cv = c.get("value", 0)
                cu = c.get("unit", "")
                ct = c.get("target", 0)
                cs = c.get("status", "amber")
                cb = c.get("benchmark", "")
                content += f"""
                <div class="action-item" style="border-left-color:{_status_color(cs)};">
                    <div class="action-item-title" style="color:{_status_color(cs)};">{cm}</div>
                    <div class="action-item-desc">{cv}{cu} (target: {ct}{cu}) — {cb}</div>
                </div>
                """

        content += """
            </div>
        </div>

        <div class="section-title">Strategic Priorities & Action Items</div>
        """

        for p in priorities:
            priority_num = p.get("priority", 0)
            area = p.get("area", "")
            current_state = p.get("current_state", "")
            action = p.get("action", "")
            impact = p.get("impact", "")
            content += f"""
        <div class="card" style="margin-bottom:1rem;border-left:4px solid {self.KP_BLUE};">
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem;">
                <div style="background:{self.KP_BLUE};color:white;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1.1rem;">
                    {priority_num}
                </div>
                <div style="font-size:1.125rem;font-weight:700;color:{self.KP_BLUE};">{area}</div>
            </div>
            <table style="margin-top:0;">
                <tr><td style="font-weight:600;width:140px;color:#666;border:none;">Current State</td><td style="border:none;">{current_state}</td></tr>
                <tr><td style="font-weight:600;width:140px;color:#666;border:none;">Recommended Action</td><td style="border:none;">{action}</td></tr>
                <tr><td style="font-weight:600;width:140px;color:#666;border:none;">Expected Impact</td><td style="border:none;color:{self.RAG_GREEN};font-weight:600;">{impact}</td></tr>
            </table>
        </div>
            """

        kpi_pct = []
        for k in kpis:
            v = k.get("value", 0)
            t = k.get("target", 1)
            kpi_pct.append(round(min((v / t * 100) if t > 0 else 0, 150), 1))

        scripts = f"""
        <script>
            new Chart(document.getElementById('{chart_id_kpi}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(kpi_labels)},
                    datasets: [{{
                        label: '% of Target',
                        data: {json.dumps(kpi_pct)},
                        backgroundColor: {json.dumps(kpi_colors)},
                        borderRadius: 4,
                        barPercentage: 0.6
                    }},
                    {{
                        label: 'Target (100%)',
                        data: {json.dumps([100] * len(kpis))},
                        type: 'line',
                        borderColor: '#ccc',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {{
                        legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
                        tooltip: {{
                            callbacks: {{
                                label: function(ctx) {{ return ctx.dataset.label + ': ' + ctx.parsed.x.toFixed(1) + '%'; }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            beginAtZero: true,
                            max: 150,
                            ticks: {{ callback: function(v) {{ return v + '%'; }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

        insight_scripts = self._build_insight_scripts(exec_insights) if exec_insights else ""
        scripts += insight_scripts

        return self._get_base_html("KP Medicare Advantage — Executive Summary", content, scripts)

    def render_revenue_optimization(self, data: dict) -> str:
        chart_id_sources = self._get_chart_id()
        chart_id_opportunities = self._get_chart_id()

        total_opportunity = self._safe_get(data, "total_opportunity", 0)
        realized_revenue = self._safe_get(data, "realized_revenue", 0)
        unrealized_revenue = self._safe_get(data, "unrealized_revenue", 0)

        revenue_sources = self._safe_get(data, "revenue_sources", {})
        opportunities = self._safe_get(data, "opportunities", {})
        initiatives = self._safe_get(data, "initiatives", [])

        source_types = list(revenue_sources.keys()) if revenue_sources else []
        source_values = list(revenue_sources.values()) if revenue_sources else []

        opp_types = list(opportunities.keys()) if opportunities else []
        opp_values = list(opportunities.values()) if opportunities else []

        content = f"""
        <div class="grid grid-3">
            <div class="card metric-card">
                <div class="metric-label">Total Opportunity</div>
                <div class="metric-value">${self._safe_number(total_opportunity)}</div>
                <div class="metric-change positive">↑ 12.5%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">Realized Revenue</div>
                <div class="metric-value">${self._safe_number(realized_revenue)}</div>
                <div class="metric-change positive">↑ 8.3%</div>
            </div>
            <div class="card metric-card">
                <div class="metric-label">Unrealized Revenue</div>
                <div class="metric-value">${self._safe_number(unrealized_revenue)}</div>
                <div class="metric-change negative">↑ 15.2%</div>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="chart-title">Revenue Sources</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_sources}"></canvas>
                </div>
            </div>

            <div class="card">
                <div class="chart-title">Optimization Opportunities</div>
                <div class="chart-container-large">
                    <canvas id="{chart_id_opportunities}"></canvas>
                </div>
            </div>
        </div>

        <div class="card card-full">
            <div class="chart-title">Active Initiatives</div>
            <table>
                <thead>
                    <tr>
                        <th>Initiative</th>
                        <th>Expected Impact</th>
                        <th>Timeline</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
        """

        for initiative in initiatives:
            initiative_name = self._safe_get(initiative, "name", "Unknown")
            impact = self._safe_get(initiative, "expected_impact", 0)
            timeline = self._safe_get(initiative, "timeline", "TBD")
            status = self._safe_get(initiative, "status", "YELLOW")

            dot_class = "green" if status == "ON TRACK" else ("yellow" if status == "AT RISK" else "red")

            content += f"""
                    <tr>
                        <td><strong>{initiative_name}</strong></td>
                        <td>${self._safe_number(impact)}</td>
                        <td>{timeline}</td>
                        <td><span class="status-dot {dot_class}"></span> {status}</td>
                    </tr>
            """

        content += """
                </tbody>
            </table>
        </div>
        """

        scripts = f"""
        <script>
            new Chart(document.getElementById('{chart_id_sources}'), {{
                type: 'pie',
                data: {{
                    labels: {json.dumps(source_types)},
                    datasets: [{{
                        data: {json.dumps(source_values)},
                        backgroundColor: ['{self.KP_BLUE}', '#4ECDC4', '#FFD93D', '#FF6B6B'],
                        borderColor: '{self.KP_WHITE}',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{ font: {{ size: 11 }} }}
                        }}
                    }}
                }}
            }});

            new Chart(document.getElementById('{chart_id_opportunities}'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(opp_types)},
                    datasets: [{{
                        label: 'Revenue Opportunity ($)',
                        data: {json.dumps(opp_values)},
                        backgroundColor: '{self.KP_BLUE}',
                        borderRadius: 4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ font: {{ size: 12, weight: 'bold' }} }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            ticks: {{ callback: function(value) {{ return '$' + (value / 1000).toFixed(0) + 'K'; }} }}
                        }}
                    }}
                }}
            }});
        </script>
        """

        return self._get_base_html("Revenue Optimization Dashboard", content, scripts)
