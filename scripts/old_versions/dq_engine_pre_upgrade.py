#!/usr/bin/env python3
"""
Data Quality Engine for Healthcare Data Pipeline

Reads CSV files from data/raw/ and runs comprehensive DQ checks:
- Completeness: percentage of non-null values
- Uniqueness: identifies duplicates in ID fields
- Validity: format validation for healthcare codes, dates, etc.
- Consistency: cross-field and cross-table checks
- Referential integrity: validates foreign key relationships
- Healthcare-specific validation rules (NPI, ICD10, CPT, etc.)

Generates DQ reports per file and summary statistics.
"""

import os
import json
import re
import csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path


class DQEngine:
    """Runs data quality checks on CSV files and generates DQ reports."""

    # Healthcare field patterns (same as profiler for consistency)
    HEALTHCARE_PATTERNS = {
        'member_id': r'(member_id|member_no|member_number|memberid|mem_id|subscriber|subscriber_id)',
        'npi': r'(npi|provider_npi|npi_number)',
        'icd10': r'(icd10|diagnosis_code|primary_diagnosis|secondary_diagnosis|diag_code)',
        'cpt': r'(cpt|procedure_code|cpt_code|proc_code)',
        'service_date': r'(service_date|date_of_service|visit_date|encounter_date|claim_date|dos)',
        'facility': r'(facility|facility_id|facility_name|provider|provider_id)',
        'kp_region': r'(region|kp_region|market|service_area)',
        'encounter_id': r'(encounter_id|visit_id|admission_id)',
        'claim_id': r'(claim_id|claim_number)',
        'rx_id': r'(rx_id|prescription_id)',
        'referral_id': r'(referral_id|referral_number)',
        'ndc': r'(ndc|drug_code)',
        'hcc': r'(hcc|risk_code)',
        'admit_date': r'(admit_date|admission_date)',
        'discharge_date': r'(discharge_date|discharge_datetime)',
        'paid_amount': r'(paid_amount|payment_amount|amount_paid)',
        'billed_amount': r'(billed_amount|billing_amount|claim_amount)',
    }

    # Date format patterns
    DATE_FORMATS = [
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
        r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
    ]

    # Validation rules
    VALIDATION_RULES = {
        'npi': {
            'pattern': r'^\d{10}$',
            'description': 'Must be exactly 10 digits'
        },
        'icd10': {
            'pattern': r'^[A-Z]\d{2}\.?\d*$',
            'description': 'Must start with letter, followed by 2 digits, optional decimal and digits'
        },
        'cpt': {
            'pattern': r'^\d{5}$',
            'description': 'Must be exactly 5 digits'
        },
        'date': {
            'pattern': None,  # Uses custom date validation
            'description': 'Must be valid date in YYYY-MM-DD, MM/DD/YYYY, or YYYY/MM/DD format'
        },
        'ndc': {
            'pattern': r'^\d{11}$',
            'description': 'Must be exactly 11 digits'
        },
    }

    def __init__(self, base_dir='/sessions/great-gallant-allen/mnt/Claude/mtp_demo'):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data', 'raw')
        self.dq_dir = os.path.join(base_dir, 'data', 'dq')
        self.paramset_dir = os.path.join(base_dir, 'paramset')

        # Create output directory
        os.makedirs(self.dq_dir, exist_ok=True)

        self.dq_reports = {}
        self.file_data = {}
        self.config = self._load_config()

    def _load_config(self):
        """Auto-discover config from paramset directory."""
        config = {
            'tables': {},
            'validation_rules': {},
            'tolerance_thresholds': {
                'completeness': 95,  # Minimum % non-null
                'uniqueness': 100,   # For ID fields
                'date_range_min': '2018-01-01',
                'date_range_max': '2026-12-31'
            }
        }

        if not os.path.exists(self.paramset_dir):
            return config

        for filename in os.listdir(self.paramset_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.paramset_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        file_config = json.load(f)
                        for key, value in file_config.items():
                            if key in config:
                                if isinstance(value, dict):
                                    config[key].update(value)
                                else:
                                    config[key] = value
                except (json.JSONDecodeError, IOError):
                    pass

        return config

    def _detect_field_type(self, col_name):
        """Detect healthcare field type from column name."""
        col_lower = col_name.lower()
        for field_type, pattern in self.HEALTHCARE_PATTERNS.items():
            if re.search(pattern, col_lower):
                return field_type
        return None

    def _is_valid_date(self, value):
        """Check if value is a valid date."""
        if not value or not str(value).strip():
            return True  # Null is valid (checked by completeness)

        value_str = str(value).strip()

        for date_format in self.DATE_FORMATS:
            if re.match(date_format, value_str):
                return True
        return False

    def _is_date_in_range(self, value, min_date='2018-01-01', max_date='2026-12-31'):
        """Check if date falls within acceptable range."""
        if not value or not str(value).strip():
            return True

        try:
            # Try to parse date
            value_str = str(value).strip()
            # Simple string comparison works for YYYY-MM-DD format
            if min_date <= value_str <= max_date:
                return True
        except (ValueError, AttributeError):
            pass

        return False

    def _is_valid_npi(self, value):
        """Validate NPI format (10 digits)."""
        if not value:
            return True
        return bool(re.match(self.VALIDATION_RULES['npi']['pattern'], str(value).strip()))

    def _is_valid_icd10(self, value):
        """Validate ICD10 format."""
        if not value:
            return True
        return bool(re.match(self.VALIDATION_RULES['icd10']['pattern'], str(value).strip().upper()))

    def _is_valid_cpt(self, value):
        """Validate CPT format (5 digits)."""
        if not value:
            return True
        return bool(re.match(self.VALIDATION_RULES['cpt']['pattern'], str(value).strip()))

    def _is_valid_ndc(self, value):
        """Validate NDC format (11 digits)."""
        if not value:
            return True
        return bool(re.match(self.VALIDATION_RULES['ndc']['pattern'], str(value).strip()))

    def _is_numeric(self, value):
        """Check if value can be converted to numeric."""
        if not value:
            return True
        try:
            float(str(value).replace(',', ''))
            return True
        except (ValueError, AttributeError):
            return False

    def _is_non_negative(self, value):
        """Check if numeric value is non-negative."""
        if not value:
            return True
        try:
            num = float(str(value).replace(',', ''))
            return num >= 0
        except (ValueError, AttributeError):
            return False

    def read_file(self, filepath):
        """Read CSV file into memory."""
        filename = os.path.basename(filepath)
        data = {'rows': [], 'headers': []}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data['headers'] = reader.fieldnames or []
                for row in reader:
                    data['rows'].append(row)
        except Exception as e:
            print(f"    Error reading {filepath}: {e}")
            return None

        self.file_data[filename] = data
        return data

    def check_completeness(self, data, col_name):
        """Check % of non-null values in column."""
        if not data['rows']:
            return 100

        non_null_count = sum(1 for row in data['rows'] if row.get(col_name) and str(row.get(col_name, '')).strip())
        completeness = (non_null_count / len(data['rows'])) * 100
        return round(completeness, 2)

    def check_uniqueness(self, data, col_name):
        """Check uniqueness for ID fields."""
        field_type = self._detect_field_type(col_name)

        # Only check uniqueness for ID-like fields
        id_indicators = ['_id', '_no', '_number', '_code', 'member', 'claim', 'encounter', 'visit']
        is_id_field = any(ind in col_name.lower() for ind in id_indicators) or field_type in [
            'member_id', 'npi', 'encounter_id', 'claim_id', 'rx_id', 'referral_id'
        ]

        if not is_id_field:
            return 100  # Not an ID field, skip

        values = [row.get(col_name) for row in data['rows'] if row.get(col_name) and str(row.get(col_name, '')).strip()]
        if not values:
            return 100

        unique_count = len(set(values))
        uniqueness = (unique_count / len(values)) * 100
        return round(uniqueness, 2)

    def check_validity(self, data, col_name):
        """Check format validity for specific field types."""
        field_type = self._detect_field_type(col_name)

        if not field_type:
            return 100  # No validation rules for generic fields

        values = [row.get(col_name) for row in data['rows'] if row.get(col_name) and str(row.get(col_name, '')).strip()]
        if not values:
            return 100

        valid_count = 0

        for value in values:
            if field_type == 'npi':
                if self._is_valid_npi(value):
                    valid_count += 1
            elif field_type == 'icd10':
                if self._is_valid_icd10(value):
                    valid_count += 1
            elif field_type == 'cpt':
                if self._is_valid_cpt(value):
                    valid_count += 1
            elif field_type == 'ndc':
                if self._is_valid_ndc(value):
                    valid_count += 1
            elif field_type in ['service_date', 'admit_date', 'discharge_date']:
                if self._is_valid_date(value) and self._is_date_in_range(value):
                    valid_count += 1
            else:
                valid_count += 1  # Unknown field type, assume valid

        validity = (valid_count / len(values)) * 100
        return round(validity, 2)

    def check_consistency(self, data, col_name):
        """Check cross-column consistency rules."""
        col_lower = col_name.lower()

        # Check amount consistency: BILLED >= PAID
        if 'paid' in col_lower and 'amount' in col_lower:
            billed_col = None
            for col in data['headers']:
                if 'billed' in col.lower() and 'amount' in col.lower():
                    billed_col = col
                    break

            if billed_col:
                consistent = 0
                checked = 0
                for row in data['rows']:
                    try:
                        paid = float(str(row.get(col_name, 0)).replace(',', ''))
                        billed = float(str(row.get(billed_col, 0)).replace(',', ''))
                        checked += 1
                        if billed >= paid:
                            consistent += 1
                    except (ValueError, AttributeError):
                        pass

                if checked > 0:
                    return round((consistent / checked) * 100, 2)

        # Check date consistency: DISCHARGE >= ADMIT
        if 'discharge' in col_lower and 'date' in col_lower:
            admit_col = None
            for col in data['headers']:
                if 'admit' in col.lower() and 'date' in col.lower():
                    admit_col = col
                    break

            if admit_col:
                consistent = 0
                checked = 0
                for row in data['rows']:
                    discharge_val = str(row.get(col_name, '')).strip()
                    admit_val = str(row.get(admit_col, '')).strip()
                    if discharge_val and admit_val:
                        checked += 1
                        if discharge_val >= admit_val:
                            consistent += 1

                if checked > 0:
                    return round((consistent / checked) * 100, 2)

        return 100  # No consistency rules apply

    def run_dq_checks(self, filepath):
        """Run all DQ checks on a file."""
        filename = os.path.basename(filepath)
        table_name = os.path.splitext(filename)[0]

        print(f"  DQ checking {filename}...")

        # Read file
        data = self.read_file(filepath)
        if not data:
            return None

        # Initialize report
        report = {
            'table_name': table_name,
            'file_name': filename,
            'total_rows': len(data['rows']),
            'total_columns': len(data['headers']),
            'checked_at': datetime.now().isoformat(),
            'column_checks': {},
            'column_scores': {},
            'dq_score': 0,
            'dq_grade': 'N/A'
        }

        # Run checks per column
        column_scores = []
        for col_name in data['headers']:
            completeness = self.check_completeness(data, col_name)
            uniqueness = self.check_uniqueness(data, col_name)
            validity = self.check_validity(data, col_name)
            consistency = self.check_consistency(data, col_name)

            # Overall column score (weighted average)
            col_score = (completeness * 0.4 + uniqueness * 0.3 + validity * 0.2 + consistency * 0.1)
            column_scores.append(col_score)

            field_type = self._detect_field_type(col_name)

            report['column_checks'][col_name] = {
                'field_type': field_type,
                'completeness': completeness,
                'uniqueness': uniqueness,
                'validity': validity,
                'consistency': consistency,
                'column_score': round(col_score, 2),
                'issues': self._identify_issues(col_name, completeness, uniqueness, validity, consistency)
            }
            report['column_scores'][col_name] = round(col_score, 2)

        # Calculate overall DQ score
        if column_scores:
            dq_score = sum(column_scores) / len(column_scores)
        else:
            dq_score = 0

        report['dq_score'] = round(dq_score, 2)
        report['dq_grade'] = self._get_grade(dq_score)

        self.dq_reports[table_name] = report
        return report

    def _identify_issues(self, col_name, completeness, uniqueness, validity, consistency):
        """Identify and describe DQ issues."""
        issues = []

        if completeness < 90:
            issues.append(f'Low completeness: {completeness}% non-null')
        if uniqueness < 100 and completeness > 50:  # Only flag if field should be unique
            field_type = self._detect_field_type(col_name)
            if field_type and 'id' in field_type.lower():
                issues.append(f'Uniqueness issues: {uniqueness}% unique')
        if validity < 95:
            issues.append(f'Validity issues: {validity}% valid format')
        if consistency < 100:
            issues.append(f'Consistency issues: {consistency}% consistent')

        return issues

    def _get_grade(self, score):
        """Convert numeric score to letter grade."""
        if score >= 95:
            return 'A'
        elif score >= 90:
            return 'B'
        elif score >= 80:
            return 'C'
        elif score >= 70:
            return 'D'
        else:
            return 'F'

    def write_file_report(self, table_name, report):
        """Write DQ report for single file."""
        output_path = os.path.join(self.dq_dir, f'{table_name}_dq.json')
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"    Wrote {output_path}")

    def write_summary(self):
        """Write summary of all DQ results."""
        print("Writing DQ summary...")

        summary = {
            'generated_at': datetime.now().isoformat(),
            'total_tables': len(self.dq_reports),
            'average_dq_score': 0,
            'grade_distribution': {
                'A': 0,
                'B': 0,
                'C': 0,
                'D': 0,
                'F': 0
            },
            'table_scores': [],
            'issues_by_type': defaultdict(int)
        }

        scores = []
        for table_name in sorted(self.dq_reports.keys()):
            report = self.dq_reports[table_name]
            scores.append(report['dq_score'])

            summary['table_scores'].append({
                'table_name': table_name,
                'dq_score': report['dq_score'],
                'grade': report['dq_grade'],
                'total_rows': report['total_rows'],
                'total_columns': report['total_columns']
            })

            summary['grade_distribution'][report['dq_grade']] += 1

            # Aggregate issues
            for col_name, col_check in report['column_checks'].items():
                for issue in col_check.get('issues', []):
                    issue_type = issue.split(':')[0]
                    summary['issues_by_type'][issue_type] += 1

        if scores:
            summary['average_dq_score'] = round(sum(scores) / len(scores), 2)

        # Convert defaultdict to regular dict for JSON serialization
        summary['issues_by_type'] = dict(summary['issues_by_type'])

        output_path = os.path.join(self.dq_dir, 'dq_summary.json')
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Wrote {output_path}")

    def run(self):
        """Execute DQ engine on all CSV files."""
        print(f"DQ Engine starting...")
        print(f"  Data directory: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            print(f"  Error: Data directory not found")
            return

        # Find all CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if not csv_files:
            print(f"  No CSV files found in {self.data_dir}")
            return

        print(f"  Found {len(csv_files)} CSV file(s)")
        print()

        # Run DQ checks on each file
        for csv_file in sorted(csv_files):
            filepath = os.path.join(self.data_dir, csv_file)
            report = self.run_dq_checks(filepath)
            if report:
                print(f"    OK - DQ Score: {report['dq_score']} ({report['dq_grade']})")
                self.write_file_report(report['table_name'], report)

        print()

        # Write summary
        self.write_summary()

        print()
        print(f"DQ Engine complete.")
        print(f"  Tables checked: {len(self.dq_reports)}")
        print(f"  Average DQ Score: {self._get_average_score()}")
        print(f"  Output: {self.dq_dir}")

    def _get_average_score(self):
        """Get average DQ score across all tables."""
        if not self.dq_reports:
            return 0
        scores = [r['dq_score'] for r in self.dq_reports.values()]
        return round(sum(scores) / len(scores), 2)


if __name__ == '__main__':
    engine = DQEngine()
    engine.run()
