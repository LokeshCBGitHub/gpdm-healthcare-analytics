"""
Deep data lineage tracking for file and column lifecycles.
Standalone module with Pure Python - no external dependencies beyond standard library.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class FileLineageNode:
    """Represents a single file in the data pipeline."""
    file_path: str
    arrival_time: str = ""
    file_size: int = 0
    row_count: int = 0
    status: str = "unknown"  # success, error, reprocessing
    errors: List[str] = field(default_factory=list)
    reprocess_history: List[Dict[str, Any]] = field(default_factory=list)
    stages: List[str] = field(default_factory=list)  # ingestion, validation, transform, etc
    dq_score: float = 0.0  # 0-100
    dq_issues: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    target_table: str = ""
    category: str = ""  # patient, claim, provider, etc


@dataclass
class ColumnLineageNode:
    """Represents a column's lineage and significance."""
    column_name: str
    source_table: str
    data_type: str = "unknown"  # varchar, int, date, numeric
    semantic_type: str = "generic"  # identifier, category, date, numeric, text
    healthcare_type: str = ""  # mrn, member_id, npi, icd10, cpt, service_date, etc
    is_phi: bool = False
    upstream: List[str] = field(default_factory=list)  # source columns
    downstream: List[str] = field(default_factory=list)  # dependent columns
    significance: str = "low"  # low, medium, high, critical
    business_value: str = ""
    usage_suggestions: List[str] = field(default_factory=list)


# Pattern matching for lineage questions
LINEAGE_PATTERNS = {
    "file_lifecycle": [
        r"when\s+did\s+.*arrive",
        r"when\s+was\s+.*\s+received",
        r"history\s+of\s+.*file",
        r"lifecycle\s+of\s+.*",
        r"where\s+did\s+.*go",
        r"what\s+happened\s+to\s+.*file",
        r"file.*status\s+of\s+.*",
    ],
    "file_errors": [
        r"errors\s+with\s+.*file",
        r"what.*wrong\s+with\s+.*",
        r"issues\s+in\s+.*file",
        r"failed.*file",
        r"problems\s+with\s+.*",
    ],
    "file_reprocessing": [
        r"reprocess.*",
        r"rerun.*file",
        r"retry\s+.*",
        r"processed.*again",
    ],
    "column_significance": [
        r"where\s+is\s+.*\s+used",
        r"significance\s+of\s+.*",
        r"what\s+(business\s+)?value.*bring",
        r"how\s+is\s+.*\s+used",
        r"role\s+of\s+.*\s+column",
        r"what\s+does\s+.*\s+mean",
    ],
    "column_upstream": [
        r"where\s+does\s+.*\s+come\s+from",
        r"source\s+of\s+.*",
        r"upstream\s+.*",
        r"original\s+.*",
    ],
    "column_downstream": [
        r"what\s+uses\s+.*",
        r"downstream\s+.*",
        r"depends\s+on\s+.*",
        r"where\s+does\s+.*\s+flow",
    ],
    "impact_analysis": [
        r"if\s+.*\s+changes\s+what\s+(breaks|happens)",
        r"impact\s+of\s+.*",
        r"what\s+breaks\s+if\s+.*",
        r"affected.*if\s+.*\s+changes",
    ],
    "all_lineage": [
        r"full.*lineage",
        r"data.*lineage",
        r"complete.*lineage",
        r"lineage.*all",
    ],
    "all_errors": [
        r"all\s+errors",
        r"all\s+issues",
        r"what\s+errors\s+exist",
        r"all\s+problems",
    ],
    "dq_summary": [
        r"data\s+quality",
        r"dq\s+score",
        r"quality\s+issues",
    ],
    "file_details": [
        r"details\s+of\s+.*file",
        r"file\s+info\s+.*",
        r"tell\s+me\s+about\s+.*file",
    ],
}

# Significance rules for healthcare columns
SIGNIFICANCE_RULES = {
    "mrn": {
        "significance": "critical",
        "business_value": "Patient identifier for deduplication and longitudinal tracking",
        "usage_suggestions": [
            "Use for patient deduplication and identity resolution",
            "Link claims to patient history for longitudinal analysis",
            "Identify duplicate records within datasets",
            "Enable cross-system patient matching",
            "Support HIPAA-compliant patient privacy controls",
        ]
    },
    "member_id": {
        "significance": "critical",
        "business_value": "Plan member identifier for enrollment, attribution, PMPM calculation",
        "usage_suggestions": [
            "Calculate PMPM (Per Member Per Month) costs",
            "Track plan member retention and churn",
            "Attribute costs and utilization by member",
            "Enable member-level analytics",
            "Support enrollment lifecycle tracking",
        ]
    },
    "npi": {
        "significance": "high",
        "business_value": "Provider identifier for referral networks, provider performance, care continuity",
        "usage_suggestions": [
            "Analyze referral networks and provider relationships",
            "Measure provider performance metrics",
            "Identify care coordination patterns",
            "Track provider specialties and specialization",
            "Support in-network vs out-of-network analysis",
        ]
    },
    "icd10": {
        "significance": "high",
        "business_value": "Diagnosis codes for HCC risk scoring, disease burden, population health",
        "usage_suggestions": [
            "Calculate HCC (Hierarchical Condition Category) risk scores",
            "Assess disease burden and prevalence in population",
            "Identify high-risk conditions for intervention",
            "Support chronic disease management programs",
            "Enable disease-specific cohort analysis",
        ]
    },
    "cpt": {
        "significance": "high",
        "business_value": "Procedure codes for utilization analysis, cost tracking, care patterns",
        "usage_suggestions": [
            "Analyze service utilization patterns",
            "Track high-cost procedure volumes",
            "Identify overutilization and underutilization",
            "Support specialty-specific performance metrics",
            "Enable evidence-based care guidelines",
        ]
    },
    "service_date": {
        "significance": "high",
        "business_value": "Temporal dimension for trend analysis, seasonality, care timeliness",
        "usage_suggestions": [
            "Analyze temporal trends in utilization and costs",
            "Identify seasonal patterns in care delivery",
            "Measure time-to-care and care delays",
            "Support cohort retention and follow-up timing",
            "Enable longitudinal outcome tracking",
        ]
    },
    "encounter_id": {
        "significance": "high",
        "business_value": "Visit identifier for care continuity, episode grouping, episode cost",
        "usage_suggestions": [
            "Track care episodes and continuity",
            "Group services within care episodes",
            "Calculate episode-based costs",
            "Identify high-utilization visit patterns",
            "Support care coordination metrics",
        ]
    },
    "kp_region": {
        "significance": "medium",
        "business_value": "Geographic dimension for regional performance, market analysis",
        "usage_suggestions": [
            "Compare regional utilization and cost patterns",
            "Identify geographic health disparities",
            "Support regional market strategy",
            "Enable regional staffing and resource planning",
            "Track regional quality and outcome metrics",
        ]
    },
    "facility": {
        "significance": "medium",
        "business_value": "Site identifier for facility performance, capacity, care model analysis",
        "usage_suggestions": [
            "Compare facility-level performance metrics",
            "Analyze facility capacity and utilization",
            "Support facility-level quality improvement",
            "Identify high-performing and underperforming sites",
            "Enable facility-based cost and outcome analysis",
        ]
    },
}

GENERIC_SIGNIFICANCE = {
    "identifier": {
        "significance": "high",
        "business_value": "Key identifier for entity linking and deduplication",
        "usage_suggestions": ["Deduplication", "Entity resolution", "Linking datasets"]
    },
    "category": {
        "significance": "medium",
        "business_value": "Categorical grouping for segmentation",
        "usage_suggestions": ["Segmentation", "Stratification", "Group analysis"]
    },
    "date": {
        "significance": "high",
        "business_value": "Temporal dimension for trend analysis",
        "usage_suggestions": ["Trend analysis", "Seasonal analysis", "Cohort tracking"]
    },
    "numeric": {
        "significance": "medium",
        "business_value": "Quantitative measure for aggregation",
        "usage_suggestions": ["Aggregation", "Financial analysis", "Utilization tracking"]
    },
    "text": {
        "significance": "low",
        "business_value": "Descriptive field for context",
        "usage_suggestions": ["Narrative analysis", "Context", "Description"]
    },
}


class LineageTracker:
    """
    Deep data lineage tracking system for file and column lifecycles.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.config = cfg or {}
        self.file_nodes: Dict[str, FileLineageNode] = {}
        self.column_nodes: Dict[str, ColumnLineageNode] = {}
        self.column_to_files: Dict[str, List[str]] = {}  # column -> list of file_paths
        self.created_at = datetime.now().isoformat()

    def ingest_from_catalog(self, catalog: Any) -> None:
        """
        Build lineage from semantic catalog object.
        Expects catalog with .tables attribute containing table profiles.
        """
        if not hasattr(catalog, "tables"):
            return

        for table in catalog.tables:
            table_name = getattr(table, "table_name", "unknown")

            if hasattr(table, "columns"):
                for col in table.columns:
                    col_name = getattr(col, "column_name", "unknown")
                    col_key = f"{table_name}.{col_name}"

                    # Determine healthcare type
                    healthcare_type = ""
                    semantic_type = getattr(col, "semantic_type", "generic")
                    if hasattr(col, "healthcare_type"):
                        healthcare_type = col.healthcare_type

                    # Look up significance rules
                    sig_rules = SIGNIFICANCE_RULES.get(
                        healthcare_type, GENERIC_SIGNIFICANCE.get(semantic_type, {})
                    )

                    node = ColumnLineageNode(
                        column_name=col_name,
                        source_table=table_name,
                        data_type=getattr(col, "data_type", "unknown"),
                        semantic_type=semantic_type,
                        healthcare_type=healthcare_type,
                        is_phi=getattr(col, "is_phi", False),
                        upstream=getattr(col, "upstream_columns", []),
                        downstream=getattr(col, "downstream_columns", []),
                        significance=sig_rules.get("significance", "low"),
                        business_value=sig_rules.get("business_value", ""),
                        usage_suggestions=sig_rules.get("usage_suggestions", []),
                    )
                    self.column_nodes[col_key] = node

    def ingest_from_directories(self, base_dir: str) -> None:
        """
        Scan staging/, processed/, dq/, recon/ directories for file artifacts.
        """
        base_path = Path(base_dir)
        subdirs = ["staging", "processed", "dq", "recon"]

        for subdir in subdirs:
            subdir_path = base_path / subdir
            if not subdir_path.exists():
                continue

            for file_path in subdir_path.glob("**/*"):
                if file_path.is_file():
                    self._create_file_node_from_path(str(file_path), subdir)

    def _create_file_node_from_path(self, file_path: str, stage: str) -> None:
        """Create a FileLineageNode from a physical file."""
        path_obj = Path(file_path)

        if not path_obj.exists():
            return

        stat = path_obj.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Infer row count for CSV files
        row_count = 0
        if file_path.endswith(".csv"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    row_count = sum(1 for _ in f) - 1  # exclude header
            except Exception:
                pass

        node = FileLineageNode(
            file_path=file_path,
            arrival_time=mod_time,
            file_size=stat.st_size,
            row_count=row_count,
            status="success",
            stages=[stage],
            category=self._infer_category(path_obj.name),
        )

        self.file_nodes[file_path] = node

    def _infer_category(self, filename: str) -> str:
        """Infer file category from filename."""
        name_lower = filename.lower()
        if "member" in name_lower or "patient" in name_lower:
            return "patient"
        elif "claim" in name_lower:
            return "claim"
        elif "provider" in name_lower:
            return "provider"
        elif "encounter" in name_lower or "visit" in name_lower:
            return "encounter"
        else:
            return "unknown"

    def is_lineage_question(self, question: str) -> Optional[str]:
        """Classify question type; return question type or None."""
        question_lower = question.lower()
        for question_type, patterns in LINEAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return question_type
        return None

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a lineage question.
        Returns {answer: str, result_data: list}
        """
        question_type = self.is_lineage_question(question)

        if question_type == "file_lifecycle":
            file_hint = self._extract_file_hint(question)
            return self.answer_file_lifecycle(file_hint)
        elif question_type == "file_errors":
            file_hint = self._extract_file_hint(question)
            return self.answer_file_errors(file_hint)
        elif question_type == "column_significance":
            col_hint = self._extract_column_hint(question)
            return self.answer_column_significance(col_hint[0], col_hint[1])
        elif question_type == "impact_analysis":
            col_hint = self._extract_column_hint(question)
            return self.get_impact_analysis(col_hint[0])
        elif question_type == "all_lineage":
            return self.answer_all_lineage_summary()
        elif question_type == "all_errors":
            return self.answer_errors_and_issues()
        else:
            return {"answer": "Unable to classify question as lineage-related.", "result_data": []}

    def _extract_file_hint(self, question: str) -> str:
        """Extract filename hint from question."""
        # Look for filenames like patient_2026.csv, claims_march.parquet, etc
        match = re.search(r"(\w+[._]?\w*\.(?:csv|parquet|json))", question, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    def _extract_column_hint(self, question: str) -> Tuple[str, Optional[str]]:
        """Extract column and optional table hint from question."""
        # Look for quoted column names or common patterns
        match = re.search(r"['\"](\w+)['\"]", question)
        col_name = match.group(1) if match else ""

        table_hint = None
        if "." in question:
            table_match = re.search(r"(\w+)\.(\w+)", question)
            if table_match:
                table_hint = table_match.group(1)
                col_name = table_match.group(2)

        return (col_name, table_hint)

    def answer_file_lifecycle(self, file_hint: str) -> Dict[str, Any]:
        """File arrival, stages, target, errors, DQ, reprocessing."""
        file_node = self.get_file_lineage(file_hint)

        if not file_node:
            return {
                "answer": f"No file found matching '{file_hint}'.",
                "result_data": []
            }

        result = {
            "answer": f"File lifecycle for {file_node.file_path}:\n"
                      f"  Arrival: {file_node.arrival_time}\n"
                      f"  Status: {file_node.status}\n"
                      f"  Size: {file_node.file_size} bytes, {file_node.row_count} rows\n"
                      f"  Stages: {', '.join(file_node.stages)}\n"
                      f"  DQ Score: {file_node.dq_score:.1f}/100\n"
                      f"  Target Table: {file_node.target_table or 'Not specified'}\n"
                      f"  Errors: {len(file_node.errors)}\n"
                      f"  Reprocessing Count: {len(file_node.reprocess_history)}",
            "result_data": [asdict(file_node)]
        }
        return result

    def answer_file_errors(self, file_hint: str) -> Dict[str, Any]:
        """Errors and issues for a specific file."""
        file_node = self.get_file_lineage(file_hint)

        if not file_node:
            return {
                "answer": f"No file found matching '{file_hint}'.",
                "result_data": []
            }

        if not file_node.errors and not file_node.dq_issues:
            return {
                "answer": f"No errors or DQ issues found for {file_node.file_path}.",
                "result_data": []
            }

        error_list = []
        answer_parts = [f"Issues for {file_node.file_path}:"]

        if file_node.errors:
            answer_parts.append(f"  Processing Errors ({len(file_node.errors)}):")
            for err in file_node.errors:
                answer_parts.append(f"    - {err}")
                error_list.append({"type": "error", "message": err})

        if file_node.dq_issues:
            answer_parts.append(f"  DQ Issues ({len(file_node.dq_issues)}):")
            for issue in file_node.dq_issues:
                answer_parts.append(f"    - {issue}")
                error_list.append({"type": "dq_issue", "message": issue})

        return {
            "answer": "\n".join(answer_parts),
            "result_data": error_list
        }

    def answer_column_significance(self, column_hint: str, table_hint: Optional[str] = None) -> Dict[str, Any]:
        """Column type, PHI, significance, business value, upstream, downstream, usage suggestions."""
        matching_cols = []

        for col_key, col_node in self.column_nodes.items():
            if column_hint.lower() in col_key.lower():
                if table_hint is None or table_hint.lower() in col_key.lower():
                    matching_cols.append((col_key, col_node))

        if not matching_cols:
            return {
                "answer": f"No column found matching '{column_hint}'.",
                "result_data": []
            }

        col_key, col_node = matching_cols[0]  # Return first match

        answer_parts = [
            f"Column Significance: {col_key}",
            f"  Data Type: {col_node.data_type}",
            f"  Semantic Type: {col_node.semantic_type}",
            f"  Healthcare Type: {col_node.healthcare_type or 'N/A'}",
            f"  Is PHI: {col_node.is_phi}",
            f"  Significance: {col_node.significance.upper()}",
            f"  Business Value: {col_node.business_value}",
        ]

        if col_node.upstream:
            answer_parts.append(f"  Upstream Sources: {', '.join(col_node.upstream)}")

        if col_node.downstream:
            answer_parts.append(f"  Downstream Uses: {', '.join(col_node.downstream)}")

        if col_node.usage_suggestions:
            answer_parts.append(f"  Usage Suggestions:")
            for suggestion in col_node.usage_suggestions[:5]:
                answer_parts.append(f"    - {suggestion}")

        return {
            "answer": "\n".join(answer_parts),
            "result_data": [asdict(col_node)]
        }

    def answer_all_lineage_summary(self) -> Dict[str, Any]:
        """Overview with status breakdown."""
        status_counts = {}
        for node in self.file_nodes.values():
            status = node.status
            status_counts[status] = status_counts.get(status, 0) + 1

        answer_parts = [
            "Full Data Lineage Summary",
            f"  Files: {len(self.file_nodes)}",
            f"  Columns: {len(self.column_nodes)}",
            "  File Status Breakdown:",
        ]

        for status, count in status_counts.items():
            answer_parts.append(f"    {status}: {count}")

        total_rows = sum(node.row_count for node in self.file_nodes.values())
        total_size = sum(node.file_size for node in self.file_nodes.values())
        avg_dq = sum(node.dq_score for node in self.file_nodes.values()) / len(self.file_nodes) if self.file_nodes else 0

        answer_parts.extend([
            f"  Total Rows Ingested: {total_rows:,}",
            f"  Total Size: {total_size:,} bytes",
            f"  Average DQ Score: {avg_dq:.1f}/100",
        ])

        return {
            "answer": "\n".join(answer_parts),
            "result_data": [
                {"metric": "file_count", "value": len(self.file_nodes)},
                {"metric": "column_count", "value": len(self.column_nodes)},
                {"metric": "total_rows", "value": total_rows},
                {"metric": "total_size_bytes", "value": total_size},
                {"metric": "avg_dq_score", "value": round(avg_dq, 1)},
            ]
        }

    def answer_errors_and_issues(self) -> Dict[str, Any]:
        """All errors and DQ issues across all files."""
        all_errors = []

        for file_path, file_node in self.file_nodes.items():
            for err in file_node.errors:
                all_errors.append({
                    "file": file_path,
                    "type": "error",
                    "message": err
                })
            for issue in file_node.dq_issues:
                all_errors.append({
                    "file": file_path,
                    "type": "dq_issue",
                    "message": issue
                })

        if not all_errors:
            return {
                "answer": "No errors or DQ issues found in the system.",
                "result_data": []
            }

        answer_parts = [f"Total Issues Found: {len(all_errors)}"]
        for item in all_errors[:10]:  # Show first 10
            answer_parts.append(f"  [{item['type']}] {item['file']}: {item['message']}")

        if len(all_errors) > 10:
            answer_parts.append(f"  ... and {len(all_errors) - 10} more")

        return {
            "answer": "\n".join(answer_parts),
            "result_data": all_errors
        }

    def get_impact_analysis(self, column_hint: str) -> Dict[str, Any]:
        """BFS through lineage graph for downstream impacts."""
        matching_cols = [
            (k, v) for k, v in self.column_nodes.items()
            if column_hint.lower() in k.lower()
        ]

        if not matching_cols:
            return {
                "answer": f"No column found matching '{column_hint}'.",
                "result_data": []
            }

        col_key, col_node = matching_cols[0]

        # BFS to find all downstream impacts
        visited = set()
        queue = [col_key]
        impact_chain = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            impact_chain.append(current)

            # Find all columns downstream of current
            for other_key, other_node in self.column_nodes.items():
                if current in other_node.upstream and other_key not in visited:
                    queue.append(other_key)

        answer_parts = [
            f"Impact Analysis for {col_key}:",
            f"  Direct Downstream: {col_node.downstream}",
            f"  Full Impact Chain: {impact_chain}",
            f"  Total Downstream Columns: {len(impact_chain) - 1}",
        ]

        return {
            "answer": "\n".join(answer_parts),
            "result_data": [{"column": col, "level": i} for i, col in enumerate(impact_chain)]
        }

    def get_file_lineage(self, file_hint: str) -> Optional[FileLineageNode]:
        """Find file node by name."""
        for file_path, node in self.file_nodes.items():
            if file_hint.lower() in file_path.lower():
                return node
        return None

    def save(self, path: str) -> None:
        """Save lineage to JSON."""
        data = {
            "created_at": self.created_at,
            "config": self.config,
            "file_nodes": {k: asdict(v) for k, v in self.file_nodes.items()},
            "column_nodes": {k: asdict(v) for k, v in self.column_nodes.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, path: str) -> None:
        """Load lineage from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        self.created_at = data.get("created_at", "")
        self.config = data.get("config", {})

        self.file_nodes = {}
        for file_path, node_dict in data.get("file_nodes", {}).items():
            node_dict["errors"] = node_dict.get("errors", [])
            node_dict["reprocess_history"] = node_dict.get("reprocess_history", [])
            node_dict["stages"] = node_dict.get("stages", [])
            node_dict["dq_issues"] = node_dict.get("dq_issues", [])
            node_dict["columns"] = node_dict.get("columns", [])
            self.file_nodes[file_path] = FileLineageNode(**node_dict)

        self.column_nodes = {}
        for col_key, node_dict in data.get("column_nodes", {}).items():
            node_dict["upstream"] = node_dict.get("upstream", [])
            node_dict["downstream"] = node_dict.get("downstream", [])
            node_dict["usage_suggestions"] = node_dict.get("usage_suggestions", [])
            self.column_nodes[col_key] = ColumnLineageNode(**node_dict)


if __name__ == "__main__":
    # Demo usage
    tracker = LineageTracker()

    # Simulate ingestion
    print("LineageTracker Demo")
    print("=" * 60)

    # Create sample file nodes
    sample_file = FileLineageNode(
        file_path="/data/staging/patient_demographics_2026.csv",
        arrival_time=datetime.now().isoformat(),
        file_size=1024000,
        row_count=50000,
        status="success",
        stages=["ingestion", "validation"],
        dq_score=92.5,
        target_table="dim_patient",
        category="patient",
    )
    tracker.file_nodes[sample_file.file_path] = sample_file

    # Create sample column nodes
    sample_col = ColumnLineageNode(
        column_name="member_id",
        source_table="dim_patient",
        data_type="varchar",
        semantic_type="identifier",
        healthcare_type="member_id",
        is_phi=True,
        upstream=["raw_enrollment.member_id"],
        downstream=["fact_claims.member_id", "fact_encounters.member_id"],
        significance="critical",
        business_value="Plan member identifier for enrollment, attribution, PMPM calculation",
        usage_suggestions=SIGNIFICANCE_RULES["member_id"]["usage_suggestions"],
    )
    tracker.column_nodes["dim_patient.member_id"] = sample_col

    # Test methods
    print("\n1. File Lifecycle:")
    result = tracker.answer_file_lifecycle("patient_demographics")
    print(result["answer"])

    print("\n2. Column Significance:")
    result = tracker.answer_column_significance("member_id")
    print(result["answer"])

    print("\n3. Lineage Summary:")
    result = tracker.answer_all_lineage_summary()
    print(result["answer"])

    print("\n4. Impact Analysis:")
    result = tracker.get_impact_analysis("member_id")
    print(result["answer"])
