import os
import json
import re
import csv
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path


class SemanticProfiler:

    HEALTHCARE_PATTERNS = {
        'member_id': r'(member_id|member_no|member_number|memberid|mem_id|subscriber|subscriber_id)',
        'npi': r'(npi|provider_npi|npi_number)',
        'icd10': r'(icd10|diagnosis_code|primary_diagnosis|secondary_diagnosis|diag_code)',
        'cpt': r'(cpt|procedure_code|cpt_code|proc_code)',
        'service_date': r'(service_date|date_of_service|visit_date|encounter_date|claim_date|dos)',
        'facility': r'(facility|facility_id|facility_name|provider|provider_id|hospital)',
        'kp_region': r'(region|kp_region|market|service_area|network)',
        'encounter_id': r'(encounter_id|visit_id|admission_id|visit_number)',
        'claim_id': r'(claim_id|claim_number|claim_no)',
        'rx_id': r'(rx_id|prescription_id|script_id)',
        'referral_id': r'(referral_id|referral_number)',
        'ndc': r'(ndc|drug_code|medication_code)',
        'hcc': r'(hcc|risk_code|condition_code)',
    }

    SEMANTIC_PATTERNS = {
        'date': r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone': r'^(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})$',
        'currency': r'^\d+\.\d{2}$|^\$\d+\.\d{2}$',
    }

    PHI_KEYWORDS = {'name', 'dob', 'date_of_birth', 'ssn', 'address', 'phone', 'email', 'mrn'}

    def __init__(self, base_dir=None):
        if base_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data', 'raw')
        self.catalog_dir = os.path.join(base_dir, 'semantic_catalog')
        self.tables_dir = os.path.join(self.catalog_dir, 'tables')
        self.relationships_dir = os.path.join(self.catalog_dir, 'relationships')
        self.context_dir = os.path.join(self.catalog_dir, 'context')
        self.paramset_dir = os.path.join(base_dir, 'paramset')

        for directory in [self.tables_dir, self.relationships_dir, self.context_dir]:
            os.makedirs(directory, exist_ok=True)

        self.table_profiles = {}
        self.shared_columns = defaultdict(list)
        self.config = self._load_config()

    def _load_config(self):
        config = {
            'tables': {},
            'semantic_mappings': {},
            'validation_rules': {}
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
                                config[key].update(value)
                            else:
                                config[key] = value
                except (json.JSONDecodeError, IOError):
                    pass

        return config

    def _infer_data_type(self, values):
        if not values:
            return 'unknown'

        non_null = [v for v in values if v and str(v).strip()]
        if not non_null:
            return 'unknown'

        bool_check = all(str(v).lower() in ['true', 'false', 'yes', 'no', '0', '1'] for v in non_null[:10])
        if bool_check and len(non_null) > 0:
            return 'boolean'

        try:
            int_check = all(float(str(v).replace(',', '')) == int(float(str(v).replace(',', '')))
                          for v in non_null[:10])
            if int_check:
                return 'integer'
        except (ValueError, AttributeError):
            pass

        try:
            float_check = all(isinstance(float(str(v).replace(',', '')), float) for v in non_null[:10])
            if float_check:
                return 'float'
        except (ValueError, AttributeError):
            pass

        date_check = all(re.match(r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$', str(v).strip())
                        for v in non_null[:10] if v)
        if date_check and len(non_null) > 0:
            return 'date'

        return 'string'

    def _detect_semantic_type(self, col_name, values, data_type):
        col_lower = col_name.lower()

        for hc_type, pattern in self.HEALTHCARE_PATTERNS.items():
            if re.search(pattern, col_lower):
                return hc_type

        non_null = [str(v).strip() for v in values if v and str(v).strip()]

        if data_type == 'date':
            return 'date'

        if re.search(self.SEMANTIC_PATTERNS['email'], non_null[0] if non_null else ''):
            return 'email'

        if re.search(self.SEMANTIC_PATTERNS['phone'], non_null[0] if non_null else ''):
            return 'phone'

        if data_type == 'float' and any(re.search(r'^\$?\d+\.\d{2}$', str(v)) for v in non_null[:10]):
            return 'currency'

        if data_type == 'integer':
            return 'code'
        elif data_type == 'float':
            return 'numeric'
        elif col_lower in ['name', 'first_name', 'last_name']:
            return 'name'
        elif col_lower in ['address', 'street', 'city', 'state', 'zip']:
            return 'address'
        elif data_type in ['string']:
            if len(set(non_null[:100])) > 50:
                return 'text'
            return 'category'

        return 'text'

    def _detect_healthcare_type(self, col_name):
        col_lower = col_name.lower()
        for hc_type, pattern in self.HEALTHCARE_PATTERNS.items():
            if re.search(pattern, col_lower):
                return hc_type
        return None

    def _is_phi(self, col_name):
        col_lower = col_name.lower()
        phi_keywords = ['name', 'dob', 'date_of_birth', 'ssn', 'address', 'phone', 'email', 'mrn']
        return any(keyword in col_lower for keyword in phi_keywords)

    def _generate_description(self, col_name, semantic_type, top_values, data_type):
        col_upper = col_name.upper()

        descriptions = {
            'member_id': 'Unique health plan member identifier used for linking enrollment, claims, and encounters',
            'npi': 'National Provider Identifier - unique 10-digit code for healthcare providers',
            'icd10': 'ICD-10 diagnosis code indicating disease, condition, or health problem',
            'cpt': 'Current Procedural Terminology code describing medical procedures or services',
            'service_date': 'Date on which healthcare service was delivered',
            'facility': 'Healthcare facility where service was rendered',
            'kp_region': 'Geographic region or service area for the health plan',
            'encounter_id': 'Unique identifier for a patient-provider interaction or visit',
            'claim_id': 'Unique identifier for an insurance claim',
            'rx_id': 'Unique identifier for a prescription',
            'referral_id': 'Unique identifier for a provider referral',
            'ndc': 'National Drug Code for pharmaceutical products',
            'hcc': 'Hierarchical Condition Category for risk adjustment',
        }

        if semantic_type in descriptions:
            return descriptions[semantic_type]

        if semantic_type == 'date':
            return f'Date field: {col_upper}'
        elif semantic_type == 'email':
            return f'Email address: {col_upper}'
        elif semantic_type == 'phone':
            return f'Telephone number: {col_upper}'
        elif semantic_type == 'currency':
            return f'Monetary amount: {col_upper}'
        elif semantic_type == 'code':
            return f'Coded value: {col_upper}'
        elif semantic_type == 'name':
            return f'Person name: {col_upper}'
        elif semantic_type == 'address':
            return f'Geographic address: {col_upper}'
        elif data_type == 'integer':
            return f'Integer value: {col_upper}'
        elif data_type == 'float':
            return f'Decimal numeric value: {col_upper}'

        return f'Field: {col_upper}'

    def profile_file(self, filepath):
        filename = os.path.basename(filepath)
        table_name = os.path.splitext(filename)[0]

        print(f"  Profiling {filename}...")

        column_data = defaultdict(list)
        row_count = 0
        headers = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                for row in reader:
                    row_count += 1
                    for col, value in row.items():
                        column_data[col].append(value)
        except Exception as e:
            print(f"    Error reading {filepath}: {e}")
            return None

        column_profiles = {}
        for col_name in headers:
            values = column_data.get(col_name, [])
            data_type = self._infer_data_type(values)
            semantic_type = self._detect_semantic_type(col_name, values, data_type)
            healthcare_type = self._detect_healthcare_type(col_name)

            non_null_values = [v for v in values if v and str(v).strip()]
            null_count = len(values) - len(non_null_values)
            null_percentage = (null_count / len(values) * 100) if values else 0
            cardinality = len(set(non_null_values))

            min_val = None
            max_val = None
            mean_val = None
            if data_type in ['integer', 'float']:
                try:
                    numeric_vals = [float(str(v).replace(',', '')) for v in non_null_values if v]
                    if numeric_vals:
                        min_val = min(numeric_vals)
                        max_val = max(numeric_vals)
                        mean_val = sum(numeric_vals) / len(numeric_vals)
                except (ValueError, AttributeError):
                    pass

            top_values_list = []
            if non_null_values:
                counter = Counter(non_null_values)
                top_values_list = [
                    {'value': val, 'count': count}
                    for val, count in counter.most_common(10)
                ]

            sample_values = []
            if non_null_values:
                import random
                sample_size = min(5, len(set(non_null_values)))
                sample_values = random.sample(non_null_values, sample_size)

            description = self._generate_description(col_name, semantic_type, top_values_list, data_type)

            column_profiles[col_name] = {
                'column_name': col_name,
                'data_type': data_type,
                'semantic_type': semantic_type,
                'healthcare_type': healthcare_type,
                'is_phi': self._is_phi(col_name),
                'cardinality': cardinality,
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2),
                'min_value': min_val,
                'max_value': max_val,
                'mean': mean_val,
                'top_values': top_values_list,
                'description': description,
                'sample_values': sample_values
            }

        semantic_types = [cp['semantic_type'] for cp in column_profiles.values()]
        healthcare_types = [cp['healthcare_type'] for cp in column_profiles.values() if cp['healthcare_type']]

        table_purpose = self._detect_table_purpose(table_name, semantic_types, healthcare_types)

        table_profile = {
            'table_name': table_name,
            'file_name': filename,
            'total_rows': row_count,
            'total_columns': len(headers),
            'table_purpose': table_purpose,
            'columns': column_profiles,
            'dq_score': 85,
            'profiled_at': datetime.now().isoformat(),
            'has_phi': any(cp['is_phi'] for cp in column_profiles.values())
        }

        self.table_profiles[table_name] = table_profile

        for col in headers:
            self.shared_columns[col.lower()].append(table_name)

        return table_profile

    def _detect_table_purpose(self, table_name, semantic_types, healthcare_types):
        table_lower = table_name.lower()

        if 'member' in table_lower or 'enrollment' in table_lower:
            return 'Member enrollment and demographics'
        elif 'claim' in table_lower:
            return 'Insurance claims and billing'
        elif 'encounter' in table_lower or 'visit' in table_lower:
            return 'Patient encounters and visits'
        elif 'diagnosis' in table_lower or 'condition' in table_lower:
            return 'Diagnoses and medical conditions'
        elif 'procedure' in table_lower or 'service' in table_lower:
            return 'Medical procedures and services'
        elif 'pharmacy' in table_lower or 'rx' in table_lower or 'prescription' in table_lower:
            return 'Pharmacy and prescription records'
        elif 'provider' in table_lower or 'facility' in table_lower:
            return 'Provider and facility information'
        elif 'referral' in table_lower:
            return 'Referral and authorization records'

        if 'npi' in healthcare_types and 'claim_id' in healthcare_types:
            return 'Claims with provider information'
        elif 'icd10' in healthcare_types:
            return 'Diagnosis or condition records'
        elif 'cpt' in healthcare_types:
            return 'Procedure or service records'
        elif 'member_id' in healthcare_types:
            return 'Member-related reference data'

        return 'Data table'

    FK_COLUMNS = {
        'member_id', 'encounter_id', 'claim_id', 'diagnosis_id', 'referral_id',
        'rx_id', 'prescription_id', 'npi', 'provider_npi', 'rendering_npi',
        'prescribing_npi', 'referring_npi', 'diagnosing_npi', 'mrn',
    }

    ATTRIBUTE_COLUMNS = {
        'status', 'kp_region', 'facility', 'department', 'specialty',
        'service_date', 'copay', 'plan_type', 'visit_type', 'gender',
        'icd10_code', 'icd10_description', 'cpt_code', 'cpt_description',
        'denial_reason', 'claim_type', 'claim_status',
    }

    def detect_relationships(self):
        relationships = []

        PRIMARY_KEYS = {
            'members': 'member_id',
            'claims': 'claim_id',
            'encounters': 'encounter_id',
            'diagnoses': 'diagnosis_id',
            'prescriptions': 'rx_id',
            'providers': 'npi',
            'referrals': 'referral_id',
        }

        pk_table = {}
        for table, pk_col in PRIMARY_KEYS.items():
            pk_table[pk_col] = table

        for col_lower, tables in self.shared_columns.items():
            if len(tables) <= 1:
                continue

            if col_lower in self.ATTRIBUTE_COLUMNS:
                continue

            is_fk = col_lower in self.FK_COLUMNS or col_lower.endswith('_id') or col_lower.endswith('_npi')
            if not is_fk:
                continue

            for i, source_table in enumerate(tables):
                for target_table in tables[i+1:]:
                    owner_table = pk_table.get(col_lower)

                    if owner_table == source_table:
                        rel_type = 'one_to_many'
                    elif owner_table == target_table:
                        rel_type = 'many_to_one'
                    else:
                        src_card = self._get_cardinality(source_table, col_lower)
                        tgt_card = self._get_cardinality(target_table, col_lower)
                        src_rows = self._get_row_count(source_table)
                        tgt_rows = self._get_row_count(target_table)

                        if src_card == src_rows and tgt_card < tgt_rows:
                            rel_type = 'one_to_many'
                        elif tgt_card == tgt_rows and src_card < src_rows:
                            rel_type = 'many_to_one'
                        elif src_card < src_rows and tgt_card < tgt_rows:
                            rel_type = 'many_to_many'
                        else:
                            rel_type = 'many_to_one'

                    relationships.append({
                        'source_table': source_table,
                        'target_table': target_table,
                        'join_column': col_lower,
                        'relationship_type': rel_type,
                        'is_fk': True,
                    })

        CROSS_COLUMN_FKS = [
            ('providers', 'npi', 'claims', 'rendering_npi'),
            ('providers', 'npi', 'encounters', 'rendering_npi'),
            ('providers', 'npi', 'prescriptions', 'prescribing_npi'),
            ('providers', 'npi', 'referrals', 'referring_npi'),
            ('providers', 'npi', 'diagnoses', 'diagnosing_npi'),
        ]

        for pk_tbl, pk_col, fk_tbl, fk_col in CROSS_COLUMN_FKS:
            pk_exists = any(c.lower() == pk_col for c in
                           self.table_profiles.get(pk_tbl, {}).get('columns', {}))
            fk_exists = any(c.lower() == fk_col for c in
                           self.table_profiles.get(fk_tbl, {}).get('columns', {}))
            if pk_exists and fk_exists:
                relationships.append({
                    'source_table': fk_tbl,
                    'target_table': pk_tbl,
                    'join_column': f'{fk_col}={pk_col}',
                    'relationship_type': 'many_to_one',
                    'is_fk': True,
                })

        return relationships

    def _get_cardinality(self, table_name, col_name):
        profile = self.table_profiles.get(table_name, {})
        columns = profile.get('columns', {})
        for cname, cinfo in columns.items():
            if cname.lower() == col_name.lower():
                return cinfo.get('cardinality', 0)
        return 0

    def _get_row_count(self, table_name):
        return self.table_profiles.get(table_name, {}).get('total_rows', 0)

    def write_profiles(self):
        print("Writing semantic profiles...")

        for table_name, profile in self.table_profiles.items():
            output_path = os.path.join(self.tables_dir, f'{table_name}.json')
            with open(output_path, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
            print(f"  Wrote {output_path}")

    def write_relationships(self):
        print("Writing relationship map...")

        relationships = self.detect_relationships()
        output_path = os.path.join(self.relationships_dir, 'relationship_map.json')

        relationship_data = {
            'detected_at': datetime.now().isoformat(),
            'total_relationships': len(relationships),
            'relationships': relationships
        }

        with open(output_path, 'w') as f:
            json.dump(relationship_data, f, indent=2, default=str)
        print(f"  Wrote {output_path}")

    def write_full_context(self):
        print("Writing full context document...")

        context_lines = []
        context_lines.append("=" * 80)
        context_lines.append("SEMANTIC CATALOG - FULL CONTEXT")
        context_lines.append("=" * 80)
        context_lines.append(f"Generated: {datetime.now().isoformat()}")
        context_lines.append("")

        context_lines.append("TABLE INVENTORY")
        context_lines.append("-" * 80)
        for table_name, profile in sorted(self.table_profiles.items()):
            context_lines.append(f"\n{table_name}")
            context_lines.append(f"  Purpose: {profile['table_purpose']}")
            context_lines.append(f"  Rows: {profile['total_rows']}")
            context_lines.append(f"  Columns: {profile['total_columns']}")
            context_lines.append(f"  Contains PHI: {profile['has_phi']}")

        context_lines.append("\n" + "=" * 80)
        context_lines.append("COLUMN SPECIFICATIONS")
        context_lines.append("=" * 80)

        for table_name in sorted(self.table_profiles.keys()):
            profile = self.table_profiles[table_name]
            context_lines.append(f"\n{table_name}")
            context_lines.append("-" * 80)

            for col_name, col_profile in profile['columns'].items():
                context_lines.append(f"\n  {col_name}")
                context_lines.append(f"    Data Type: {col_profile['data_type']}")
                context_lines.append(f"    Semantic Type: {col_profile['semantic_type']}")
                if col_profile['healthcare_type']:
                    context_lines.append(f"    Healthcare Type: {col_profile['healthcare_type']}")
                context_lines.append(f"    Description: {col_profile['description']}")
                context_lines.append(f"    Cardinality: {col_profile['cardinality']}")
                context_lines.append(f"    Nullability: {col_profile['null_percentage']}% null")
                context_lines.append(f"    PHI: {col_profile['is_phi']}")

                if col_profile['top_values']:
                    tvs = ["{} ({})".format(v['value'], v['count']) for v in col_profile['top_values'][:5]]
                    context_lines.append(f"    Top Values: {', '.join(tvs)}")

        relationships = self.detect_relationships()
        if relationships:
            context_lines.append("\n" + "=" * 80)
            context_lines.append("DETECTED RELATIONSHIPS")
            context_lines.append("=" * 80)
            for rel in relationships:
                context_lines.append(f"\n  {rel['source_table']} -> {rel['target_table']}")
                context_lines.append(f"    Join Column: {rel['join_column']}")
                context_lines.append(f"    Type: {rel['relationship_type']}")

        output_path = os.path.join(self.context_dir, 'full_context.txt')
        with open(output_path, 'w') as f:
            f.write('\n'.join(context_lines))
        print(f"  Wrote {output_path}")

    def run(self):
        print(f"Semantic Profiler starting...")
        print(f"  Data directory: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            print(f"  Error: Data directory not found")
            return

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        if not csv_files:
            print(f"  No CSV files found in {self.data_dir}")
            return

        print(f"  Found {len(csv_files)} CSV file(s)")
        print()

        for csv_file in sorted(csv_files):
            filepath = os.path.join(self.data_dir, csv_file)
            profile = self.profile_file(filepath)
            if profile:
                print(f"    OK - {profile['total_rows']} rows, {profile['total_columns']} columns")

        print()

        self.write_profiles()
        self.write_relationships()
        self.write_full_context()

        print()
        print(f"Semantic Profiler complete.")
        print(f"  Profiles: {len(self.table_profiles)} tables")
        print(f"  Output: {self.catalog_dir}")


if __name__ == '__main__':
    profiler = SemanticProfiler()
    profiler.run()
