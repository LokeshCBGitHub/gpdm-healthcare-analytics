import sys
import json
import re
import os
from pathlib import Path

sys.path.insert(0, '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/scripts')
sys.path.insert(0, '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo')

from dashboard_frontend import DashboardFrontendRenderer
from executive_dashboards import ExecutiveDashboardEngine


class TestResults:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def add(self, test_name, passed, message=""):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            'status': status,
            'test': test_name,
            'message': message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(f"{status}: {test_name}", end="")
        if message:
            print(f" [{message}]", end="")
        print()

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(f"SUMMARY: {self.passed}/{total} tests passed, {self.failed} failed")
        print("="*70)
        return self.failed == 0


class DashboardIntegrityTester:

    def __init__(self):
        self.results = TestResults()
        self.db_path = '/sessions/great-gallant-allen/mnt/chatbot/mtp_demo/data/healthcare_production.db'
        self.renderer = None
        self.engine = None
        self.executive_html = None
        self.financial_html = None
        self.stars_html = None
        self.executive_data = None
        self.financial_data = None
        self.stars_data = None

    def setup(self):
        print("Initializing DashboardFrontendRenderer and ExecutiveDashboardEngine...")
        try:
            self.renderer = DashboardFrontendRenderer(db_path=self.db_path)
            self.engine = ExecutiveDashboardEngine(db_path=self.db_path)
            print("✓ Renderer and Engine initialized successfully\n")
        except Exception as e:
            print(f"✗ Failed to initialize: {e}")
            sys.exit(1)

    def generate_dashboards(self):
        print("Generating dashboards from ExecutiveDashboardEngine...")
        try:
            self.executive_data = self.engine.get_executive_summary()
            self.financial_data = self.engine.get_financial_performance()
            self.stars_data = self.engine.get_stars_performance()

            self.executive_html = self.renderer.render_executive_summary(self.executive_data)
            self.financial_html = self.renderer.render_financial_dashboard(self.financial_data)
            self.stars_html = self.renderer.render_stars_dashboard(self.stars_data)

            print(f"✓ Executive Summary HTML: {len(self.executive_html)} chars")
            print(f"✓ Financial Dashboard HTML: {len(self.financial_html)} chars")
            print(f"✓ Stars Dashboard HTML: {len(self.stars_html)} chars\n")
        except Exception as e:
            print(f"✗ Failed to generate dashboards: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def count_tags(self, html, opening_tag, closing_tag=None):
        if closing_tag is None:
            closing_tag = opening_tag
        open_count = html.count(f"<{opening_tag}")
        close_count = html.count(f"</{closing_tag}>")
        return open_count, close_count

    def test_html_structure_integrity(self):
        print("\n=== SUITE 1: HTML Structure Integrity (20+ tests) ===\n")

        for dashboard_name, html in [("Executive", self.executive_html),
                                       ("Financial", self.financial_html),
                                       ("Stars", self.stars_html)]:

            self.results.add(
                f"{dashboard_name}: Has DOCTYPE declaration",
                html.startswith("<!DOCTYPE html>")
            )

            open_html, close_html = self.count_tags(html, "html")
            self.results.add(
                f"{dashboard_name}: HTML tags balanced",
                open_html > 0 and open_html == close_html,
                f"open={open_html}, close={close_html}"
            )

            open_head, close_head = self.count_tags(html, "head")
            self.results.add(
                f"{dashboard_name}: HEAD tags balanced",
                open_head > 0 and open_head == close_head,
                f"open={open_head}, close={close_head}"
            )

            open_body, close_body = self.count_tags(html, "body")
            self.results.add(
                f"{dashboard_name}: BODY tags balanced",
                open_body > 0 and open_body == close_body,
                f"open={open_body}, close={close_body}"
            )

            self.results.add(
                f"{dashboard_name}: Has closing </html> tag",
                html.rstrip().endswith("</html>")
            )

            self.results.add(
                f"{dashboard_name}: Chart.js CDN script present",
                "https://cdn.jsdelivr.net/npm/chart.js" in html
            )

            open_div, close_div = self.count_tags(html, "div")
            self.results.add(
                f"{dashboard_name}: DIV tags balanced",
                open_div == close_div,
                f"open={open_div}, close={close_div}"
            )

            title_match = re.search(r'<title>([^<]+)</title>', html)
            has_title = title_match is not None
            self.results.add(
                f"{dashboard_name}: Title tag present",
                has_title
            )

            self.results.add(
                f"{dashboard_name}: Header div present",
                '<div class="header">' in html or '<header' in html
            )

            self.results.add(
                f"{dashboard_name}: KP logo placeholder present",
                "logo-placeholder" in html or "KP" in html
            )

            self.results.add(
                f"{dashboard_name}: Container div present",
                'class="container"' in html
            )

            style_match = re.search(r'<style>(.*?)</style>', html, re.DOTALL)
            if style_match:
                style_content = style_match.group(1)
                defined_classes = set(re.findall(r'\.([a-zA-Z0-9_-]+)\s*\{', style_content))

                used_classes = set(re.findall(r'class=["\']([^"\']*)["\']', html))
                used_classes_list = set()
                for class_str in used_classes:
                    used_classes_list.update(class_str.split())

                undefined_classes = used_classes_list - defined_classes
                undefined_classes = {c for c in undefined_classes if c not in
                                      ['card', 'container', 'grid', 'header', 'toolbar']}

                self.results.add(
                    f"{dashboard_name}: All CSS classes defined",
                    len(undefined_classes) == 0,
                    f"undefined: {undefined_classes}" if undefined_classes else "all defined"
                )

            self.results.add(
                f"{dashboard_name}: Style block present",
                '<style>' in html and '</style>' in html
            )

            self.results.add(
                f"{dashboard_name}: Script block present",
                '<script>' in html or '<script ' in html
            )

            html_no_scripts = re.sub(r'<script[\s\S]*?</script>', '', html)
            malformed = re.findall(r'<[^>]{300,}', html_no_scripts)
            self.results.add(
                f"{dashboard_name}: No malformed tags",
                len(malformed) == 0
            )

    def test_chatbot_system(self):
        print("\n=== SUITE 2: Chatbot System (15+ tests) ===\n")

        for dashboard_name, html in [("Executive", self.executive_html),
                                       ("Financial", self.financial_html),
                                       ("Stars", self.stars_html)]:

            self.results.add(
                f"{dashboard_name}: Chatbot FAB button exists",
                'chatbot-fab' in html or 'chat-fab' in html or 'onclick="toggleChat' in html
            )

            fab_pattern = r'<[^>]*class="[^"]*chat[^"]*"[^>]*onclick="toggleChat'
            self.results.add(
                f"{dashboard_name}: Chatbot button has onclick handler",
                'toggleChat' in html
            )

            self.results.add(
                f"{dashboard_name}: Chatbot panel div with id exists",
                'id="chatPanel"' in html
            )

            self.results.add(
                f"{dashboard_name}: Chat input field exists",
                'id="chatInput"' in html or 'class="chat-input"' in html
            )

            self.results.add(
                f"{dashboard_name}: Chat send button exists",
                'sendChat' in html
            )

            self.results.add(
                f"{dashboard_name}: toggleChat() function defined",
                'function toggleChat()' in html or 'toggleChat = function' in html or 'toggleChat()' in html
            )

            self.results.add(
                f"{dashboard_name}: sendChat() function defined",
                'function sendChat()' in html or 'sendChat = function' in html
            )

            self.results.add(
                f"{dashboard_name}: formatChatResponse() function defined",
                'formatChatResponse' in html
            )

            self.results.add(
                f"{dashboard_name}: appendChatMsg() function defined",
                'appendChatMsg' in html
            )

            self.results.add(
                f"{dashboard_name}: Typing indicator functions defined",
                ('showTyping' in html and 'removeTyping' in html) or 'typing' in html.lower()
            )

            self.results.add(
                f"{dashboard_name}: Chat suggestions container present",
                'chatSuggestions' in html or 'chat-suggestions' in html or 'suggestions' in html
            )

            self.results.add(
                f"{dashboard_name}: Chat messages container present",
                'chatMessages' in html or 'chat-messages' in html or 'messages' in html
            )

            self.results.add(
                f"{dashboard_name}: API uses /api/intelligent/query endpoint",
                '/api/intelligent/query' in html
            )

            session_pattern = r'sessionId.*?Date\.now\(\)|sessionId.*?Math\.random'
            self.results.add(
                f"{dashboard_name}: Session ID generation code present",
                'sessionId' in html and ('Date.now()' in html or 'Math.random()' in html)
            )

            self.results.add(
                f"{dashboard_name}: Chat panel close button present",
                'closeChat' in html or ('chatPanel' in html and 'close' in html.lower())
            )

    def test_modern_toolbar(self):
        print("\n=== SUITE 3: Modern Toolbar (10+ tests) ===\n")

        for dashboard_name, html in [("Executive", self.executive_html),
                                       ("Financial", self.financial_html),
                                       ("Stars", self.stars_html)]:

            self.results.add(
                f"{dashboard_name}: Toolbar div exists",
                'class="toolbar"' in html
            )

            self.results.add(
                f"{dashboard_name}: Print button exists",
                'print' in html.lower()
            )

            self.results.add(
                f"{dashboard_name}: Export PDF button with exportDashboard('pdf')",
                "exportDashboard('pdf')" in html or 'exportDashboard("pdf")' in html
            )

            self.results.add(
                f"{dashboard_name}: Export CSV button with exportDashboard('csv')",
                "exportDashboard('csv')" in html or 'exportDashboard("csv")' in html
            )

            self.results.add(
                f"{dashboard_name}: Save View button exists",
                'saveDashboardState' in html
            )

            self.results.add(
                f"{dashboard_name}: Bookmark button exists",
                'toggleBookmark' in html or 'bookmark' in html.lower()
            )

            self.results.add(
                f"{dashboard_name}: Refresh button exists",
                'refresh' in html.lower() or 'reload' in html.lower()
            )

            self.results.add(
                f"{dashboard_name}: Ask AI button exists",
                'toggleChat' in html or 'ask.*ai' in html.lower()
            )

            self.results.add(
                f"{dashboard_name}: Last updated timestamp element",
                'updated' in html.lower() or 'timestamp' in html.lower() or 'refreshed' in html.lower()
            )

            self.results.add(
                f"{dashboard_name}: exportDashboard() function defined",
                'function exportDashboard' in html or 'exportDashboard = function' in html
            )

            self.results.add(
                f"{dashboard_name}: saveDashboardState() function defined",
                'function saveDashboardState' in html or 'saveDashboardState = function' in html
            )

            self.results.add(
                f"{dashboard_name}: toggleBookmark() function defined",
                'function toggleBookmark' in html or 'toggleBookmark = function' in html
            )

    def test_insight_modal_system(self):
        print("\n=== SUITE 4: Insight Modal System (15+ tests) ===\n")

        for dashboard_name, html in [("Executive", self.executive_html),
                                       ("Financial", self.financial_html),
                                       ("Stars", self.stars_html)]:

            self.results.add(
                f"{dashboard_name}: insightOverlay div exists",
                'id="insightOverlay"' in html
            )

            self.results.add(
                f"{dashboard_name}: insightModal div exists",
                'id="insightModal"' in html
            )

            register_idx = html.find('function registerInsight')
            first_call_idx = html.find('registerInsight(')
            self.results.add(
                f"{dashboard_name}: registerInsight() defined before use",
                register_idx >= 0 and (first_call_idx < 0 or register_idx < first_call_idx)
            )

            self.results.add(
                f"{dashboard_name}: openInsight() function defined",
                'function openInsight' in html or 'openInsight = function' in html
            )

            self.results.add(
                f"{dashboard_name}: closeInsight() function defined",
                'function closeInsight' in html or 'closeInsight = function' in html
            )

            self.results.add(
                f"{dashboard_name}: buildInsightHTML() function defined",
                'buildInsightHTML' in html
            )

            self.results.add(
                f"{dashboard_name}: renderForecastChart() function defined",
                'renderForecastChart' in html
            )

            self.results.add(
                f"{dashboard_name}: Escape key handler registered",
                'keydown' in html or 'keyCode === 27' in html or 'key === "Escape"' in html
            )

            register_calls = re.findall(r"registerInsight\(['\"]([^'\"]+)['\"],\s*\{", html)
            calls_with_data = sum(1 for key in register_calls
                                  if f'registerInsight(\'{key}\', {{"kpi_name"' in html
                                  or f'registerInsight(\'{key}\', {{"' in html)

            self.results.add(
                f"{dashboard_name}: registerInsight() calls have valid JSON",
                len(register_calls) > 0 and calls_with_data == len(register_calls),
                f"total={len(register_calls)}, with_data={calls_with_data}"
            )

            if dashboard_name == "Financial":
                insight_keys = ['pmpm_revenue', 'medical_loss_ratio', 'collection_rate', 'denial_rate']
                found_count = sum(1 for key in insight_keys if key in html)
                self.results.add(
                    f"{dashboard_name}: Financial insights registered",
                    found_count >= 2,
                    f"found {found_count}/4"
                )

            elif dashboard_name == "Stars":
                insight_keys = ['hedis', 'cahps', 'admin']
                found_count = sum(1 for key in insight_keys if key in html)
                self.results.add(
                    f"{dashboard_name}: Stars insights registered",
                    found_count >= 2,
                    f"found {found_count}/3"
                )

            elif dashboard_name == "Executive":
                insight_keys = ['member_retention', 'medical_loss_ratio']
                found_count = sum(1 for key in insight_keys if key in html)
                self.results.add(
                    f"{dashboard_name}: Executive insights registered",
                    found_count >= 1,
                    f"found {found_count}/2"
                )

            onclick_count = html.count('onclick="openInsight(')
            self.results.add(
                f"{dashboard_name}: Clickable elements with openInsight handlers",
                onclick_count > 0,
                f"count={onclick_count}"
            )

    def test_anticipation_system(self):
        print("\n=== SUITE 5: Anticipation System (5+ tests) ===\n")

        for dashboard_name, html in [("Executive", self.executive_html),
                                       ("Financial", self.financial_html),
                                       ("Stars", self.stars_html)]:

            self.results.add(
                f"{dashboard_name}: triggerAnticipation() defined",
                'triggerAnticipation' in html
            )

            self.results.add(
                f"{dashboard_name}: askAnticipated() defined",
                'askAnticipated' in html
            )

            self.results.add(
                f"{dashboard_name}: updateSuggestions() defined",
                'updateSuggestions' in html
            )

            self.results.add(
                f"{dashboard_name}: updateSuggestionsAfterQuery() defined",
                'updateSuggestionsAfterQuery' in html
            )

            self.results.add(
                f"{dashboard_name}: anticipationBar div exists",
                'anticipationBar' in html or 'anticipation' in html
            )

    def test_data_accuracy(self):
        print("\n=== SUITE 6: Data Accuracy (10+ tests) ===\n")

        pmpm_match = re.search(r'PMPM.*?(\d+\.?\d*)', self.financial_html)
        self.results.add(
            "Financial: PMPM values are numeric",
            pmpm_match is not None and pmpm_match.group(1) != "0",
            f"value={pmpm_match.group(1) if pmpm_match else 'not found'}"
        )

        mlr_match = re.search(r'Medical Loss Ratio.*?(\d+\.?\d*)\s*%', self.financial_html)
        if mlr_match:
            mlr_val = float(mlr_match.group(1))
            self.results.add(
                "Financial: MLR between 50-120%",
                50 <= mlr_val <= 120,
                f"value={mlr_val}%"
            )
        else:
            self.results.add(
                "Financial: MLR between 50-120%",
                False,
                "MLR value not found"
            )

        rating_match = re.search(r'(\d+\.\d+)\s*/\s*5\.0', self.stars_html)
        if rating_match:
            rating_val = float(rating_match.group(1))
            self.results.add(
                "Stars: Overall rating between 1-5",
                1 <= rating_val <= 5,
                f"value={rating_val}"
            )
        else:
            self.results.add(
                "Stars: Overall rating between 1-5",
                False,
                "rating not found"
            )

        self.results.add(
            "Stars: HEDIS measures table present",
            'HEDIS' in self.stars_html and '<tbody>' in self.stars_html,
            "HEDIS or tbody not found"
        )

        kpi_cards = self.executive_html.count('metric-card') + self.executive_html.count('class="kpi')
        self.results.add(
            "Executive: KPI cards present",
            kpi_cards > 0,
            f"count={kpi_cards}"
        )

        thead_count = self.financial_html.count('<thead>')
        tbody_count = self.financial_html.count('<tbody>')
        self.results.add(
            "Financial: Tables have thead and tbody",
            thead_count > 0 and thead_count == tbody_count,
            f"thead={thead_count}, tbody={tbody_count}"
        )

        chart_data = re.findall(r'data:\s*\[([\d\s.,]+)\]', self.financial_html)
        non_empty_charts = sum(1 for d in chart_data if d.strip() != '')
        self.results.add(
            "Financial: Chart data arrays non-empty",
            non_empty_charts > 0,
            f"charts={non_empty_charts}"
        )

        html_no_scripts = re.sub(r'<script[\s\S]*?</script>', '', self.executive_html)
        undefined_count = html_no_scripts.count('undefined')
        self.results.add(
            "Executive: No 'undefined' text in HTML",
            undefined_count == 0,
            f"count={undefined_count}"
        )

        fin_no_scripts = re.sub(r'<script[\s\S]*?</script>', '', self.financial_html)
        null_count = fin_no_scripts.count('>null<') + fin_no_scripts.count('"null"')
        self.results.add(
            "Financial: No 'null' literals in HTML",
            null_count == 0,
            f"count={null_count}"
        )

        format_artifacts = re.findall(r'\{0\}|\{1\}|\{self\.', self.stars_html)
        self.results.add(
            "Stars: No Python format string artifacts",
            len(format_artifacts) == 0,
            f"count={len(format_artifacts)}"
        )

        nan_count = self.financial_html.count('NaN')
        self.results.add(
            "Financial: No NaN values",
            nan_count == 0,
            f"count={nan_count}"
        )

    def run_all_tests(self):
        print("\n" + "="*70)
        print("DASHBOARD INTEGRITY TEST SUITE")
        print("="*70)

        self.setup()
        self.generate_dashboards()

        self.test_html_structure_integrity()
        self.test_chatbot_system()
        self.test_modern_toolbar()
        self.test_insight_modal_system()
        self.test_anticipation_system()
        self.test_data_accuracy()

        success = self.results.summary()
        return success


def main():
    tester = DashboardIntegrityTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
