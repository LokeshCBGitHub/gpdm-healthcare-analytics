#!/usr/bin/env python3
"""
Healthcare Dashboards Generator
Regenerates all 16 healthcare domain dashboards using the AnalyticalIntelligence engine.
Supports multiple database sizes — pass db_path and output_dir as arguments.
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from analytical_intelligence import AnalyticalIntelligence

# Dashboard configuration: (filename, question)
DASHBOARDS = [
    ('executive_dashboard.html', 'Executive summary with KPIs, PMPM, MLR, Stars, regional scorecard'),
    ('financial_dashboard.html', 'Financial performance analysis'),
    ('demographics_dashboard.html', 'Demographic population age gender race ethnicity profile distribution breakdown'),
    ('utilization_dashboard.html', 'Utilization metrics and trends'),
    ('quality_dashboard.html', 'Quality metrics and HEDIS measures'),
    ('provider_dashboard.html', 'Provider performance analysis'),
    ('clinical_outcomes_dashboard.html', 'Clinical outcomes analysis'),
    ('claims_severity_dashboard.html', 'Claims severity analysis'),
    ('pharmacy_dashboard.html', 'Pharmacy analytics and medication insights'),
    ('referral_network_dashboard.html', 'Referral network analysis'),
    ('provider_network_dashboard.html', 'Provider network adequacy'),
    ('forecasting_dashboard.html', 'Forecasting and predictive analytics'),
    ('appointment_access_dashboard.html', 'Appointment access and scheduling'),
    ('membership_intelligence_dashboard.html', 'Membership intelligence and enrollment'),
    ('population_health_dashboard.html', 'Population health management'),
    ('revenue_cycle_dashboard.html', 'Revenue cycle management'),
]

def generate_all(db_path, dashboards_dir):
    """Generate all 16 dashboards for a given database."""
    # Verify database exists
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return 1

    # Verify dashboards directory exists
    if not os.path.exists(dashboards_dir):
        os.makedirs(dashboards_dir)
        print(f"Created dashboards directory: {dashboards_dir}")

    # Initialize the analytical intelligence engine
    print(f"Initializing AnalyticalIntelligence with db_path={db_path}")
    ai = AnalyticalIntelligence(db_path)

    # Generate each dashboard
    successful = 0
    failed = 0
    total_size = 0

    print(f"\nGenerating {len(DASHBOARDS)} dashboards into {dashboards_dir}...\n")

    for filename, question in DASHBOARDS:
        output_path = os.path.join(dashboards_dir, filename)
        print(f"  Generating {filename}...", end=' ')
        sys.stdout.flush()

        try:
            result = ai.analyze(question)
            dashboard_html = result.get('dashboard_html', '')

            if not dashboard_html:
                print("ERROR: No HTML generated")
                failed += 1
                continue

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)

            file_size = os.path.getsize(output_path)
            total_size += file_size

            if file_size < 10240:
                print(f"WARNING: Small file ({file_size} bytes)")
            else:
                print(f"OK ({file_size:,} bytes)")

            successful += 1

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback; traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"Dashboard Generation Summary — {os.path.basename(db_path)}")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(DASHBOARDS)}")
    print(f"Failed: {failed}/{len(DASHBOARDS)}")
    print(f"Total file size: {total_size:,} bytes ({total_size/1024:.1f} KB)")

    # Verify files
    file_count = sum(1 for f, _ in DASHBOARDS if os.path.exists(os.path.join(dashboards_dir, f)))
    view_sql_count = 0
    for filename, _ in DASHBOARDS:
        p = os.path.join(dashboards_dir, filename)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                if 'View SQL' in f.read():
                    view_sql_count += 1

    print(f"Files found: {file_count}/{len(DASHBOARDS)}")
    print(f"Files with 'View SQL': {view_sql_count}/{file_count}")

    return 0 if failed == 0 else 1


def main():
    """Generate dashboards — supports CLI args: [db_path] [output_dir]"""
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/healthcare_demo.db'
    dashboards_dir = sys.argv[2] if len(sys.argv) > 2 else 'dashboards'
    return generate_all(db_path, dashboards_dir)


if __name__ == '__main__':
    sys.exit(main())
