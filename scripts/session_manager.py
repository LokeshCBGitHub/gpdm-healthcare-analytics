import os
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from dashboard_generator import generate_dashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


@dataclass
class QueryRecord:
    query_id: str
    timestamp: str
    question: str
    sql: str
    result_count: int
    csv_file: str
    png_file: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    engine_mode: str = 'template'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class UserSession:

    def __init__(self, username: str, sessions_dir: str = 'user_sessions'):
        self.username = username
        self.sessions_dir = sessions_dir
        self.user_dir = os.path.join(sessions_dir, username)

        os.makedirs(self.user_dir, exist_ok=True)

        self.history_file = os.path.join(self.user_dir, 'session_history.json')
        self.stats_file = os.path.join(self.user_dir, 'stats.json')

        self._history = self._load_history()
        self._stats = self._load_stats()

        self._query_counter = len(self._history) + 1

    @property
    def user_dir(self) -> str:
        return self._user_dir

    @user_dir.setter
    def user_dir(self, value: str):
        self._user_dir = value

    def save_result(
        self,
        question: str,
        sql: str,
        result_data: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        engine_mode: str = 'template'
    ) -> Dict[str, str]:
        query_id = f"Q{self._query_counter:03d}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        query_time = datetime.now().isoformat()

        csv_file = self._save_csv(query_id, timestamp, result_data)

        png_file = None
        if DASHBOARD_AVAILABLE and result_data:
            try:
                png_file = self._generate_dashboard(
                    query_id, timestamp, question, sql, result_data, metadata
                )
            except Exception as e:
                print(f"Warning: Dashboard generation failed: {e}")

        record = QueryRecord(
            query_id=query_id,
            timestamp=query_time,
            question=question,
            sql=sql,
            result_count=len(result_data),
            csv_file=os.path.basename(csv_file),
            png_file=os.path.basename(png_file) if png_file else None,
            metadata=metadata,
            engine_mode=engine_mode
        )

        self._history.append(record.to_dict())
        self._save_history()

        self._update_stats(result_data, metadata)
        self._save_stats()

        self._query_counter += 1

        return {
            'csv': csv_file,
            'png': png_file,
            'query_id': query_id
        }

    def _save_csv(self, query_id: str, timestamp: str,
                  result_data: List[Dict[str, Any]]) -> str:
        if not result_data:
            return ''

        csv_filename = f"{query_id}_{timestamp}.csv"
        csv_path = os.path.join(self.user_dir, csv_filename)

        cols = list(result_data[0].keys())

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()

                for row in result_data:
                    clean_row = {}
                    for col in cols:
                        val = row.get(col, '')
                        if isinstance(val, dict):
                            clean_row[col] = json.dumps(val)
                        else:
                            clean_row[col] = val
                    writer.writerow(clean_row)
        except Exception as e:
            print(f"Error writing CSV: {e}")
            return ''

        return csv_path

    def _generate_dashboard(self, query_id: str, timestamp: str, question: str,
                           sql: str, result_data: List[Dict[str, Any]],
                           metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if not DASHBOARD_AVAILABLE:
            return None

        png_filename = f"{query_id}_{timestamp}.png"

        try:
            png_path = generate_dashboard(
                result_data=result_data,
                question=question,
                sql=sql,
                output_path=png_filename,
                metadata=metadata,
                output_dir=self.user_dir
            )
            return png_path
        except Exception as e:
            print(f"Warning: Failed to generate dashboard: {e}")
            return None

    def _load_history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_file):
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")
            return []

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def _load_stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.stats_file):
            return {
                'username': self.username,
                'total_queries': 0,
                'total_rows': 0,
                'tables_queried': [],
                'intents_used': [],
                'created_at': datetime.now().isoformat(),
                'last_query_at': None,
            }

        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load stats: {e}")
            return {}

    def _save_stats(self):
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            print(f"Error saving stats: {e}")

    def _update_stats(self, result_data: List[Dict[str, Any]],
                     metadata: Optional[Dict[str, Any]]):
        self._stats['total_queries'] = len(self._history) + 1
        self._stats['total_rows'] += len(result_data)
        self._stats['last_query_at'] = datetime.now().isoformat()

        if metadata and 'tables' in metadata:
            tables = metadata.get('tables', [])
            if isinstance(tables, list):
                self._stats['tables_queried'] = list(
                    set(self._stats.get('tables_queried', []) + tables)
                )

        if metadata and 'intent' in metadata:
            intent = metadata.get('intent')
            intents = self._stats.get('intents_used', [])
            if intent not in intents:
                intents.append(intent)
            self._stats['intents_used'] = intents

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history

    def get_history_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                'query_id': record.get('query_id'),
                'timestamp': record.get('timestamp'),
                'question': record.get('question'),
                'result_count': record.get('result_count'),
                'engine_mode': record.get('engine_mode'),
            }
            for record in self._history
        ]

    def get_stats(self) -> Dict[str, Any]:
        return self._stats

    def get_query_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        for record in self._history:
            if record.get('query_id') == query_id:
                return record
        return None

    def get_csv_path(self, query_id: str) -> Optional[str]:
        record = self.get_query_by_id(query_id)
        if record and record.get('csv_file'):
            return os.path.join(self.user_dir, record['csv_file'])
        return None

    def get_png_path(self, query_id: str) -> Optional[str]:
        record = self.get_query_by_id(query_id)
        if record and record.get('png_file'):
            return os.path.join(self.user_dir, record['png_file'])
        return None


if __name__ == '__main__':
    """Demo: Create sample user session with queries."""

    session = UserSession('demo_user', sessions_dir='/tmp/sessions_demo')

    print(f"Created user session: {session.user_dir}\n")

    print("Query 1: Sales by region")
    result_data_1 = [
        {'region': 'North', 'sales': 45000},
        {'region': 'South', 'sales': 52000},
        {'region': 'East', 'sales': 38000},
        {'region': 'West', 'sales': 61000},
    ]

    result_1 = session.save_result(
        question="How much sales did each region generate?",
        sql="SELECT region, SUM(sales) as sales FROM sales GROUP BY region",
        result_data=result_data_1,
        metadata={
            'tables': ['sales'],
            'intent': 'compare'
        }
    )
    print(f"  CSV: {result_1['csv']}")
    print(f"  PNG: {result_1['png']}")
    print(f"  Query ID: {result_1['query_id']}\n")

    print("Query 2: Daily revenue trend")
    result_data_2 = [
        {'date': '2024-01-01', 'revenue': 5000},
        {'date': '2024-01-02', 'revenue': 6200},
        {'date': '2024-01-03', 'revenue': 5800},
        {'date': '2024-01-04', 'revenue': 7100},
        {'date': '2024-01-05', 'revenue': 7900},
    ]

    result_2 = session.save_result(
        question="What is the revenue trend over time?",
        sql="SELECT date, SUM(revenue) as revenue FROM sales GROUP BY date ORDER BY date",
        result_data=result_data_2,
        metadata={
            'tables': ['sales'],
            'intent': 'trend'
        }
    )
    print(f"  CSV: {result_2['csv']}")
    print(f"  PNG: {result_2['png']}")
    print(f"  Query ID: {result_2['query_id']}\n")

    print("Query 3: Total revenue")
    result_data_3 = [
        {'total_revenue': 125000}
    ]

    result_3 = session.save_result(
        question="What is the total revenue?",
        sql="SELECT SUM(revenue) as total_revenue FROM sales",
        result_data=result_data_3,
        metadata={
            'tables': ['sales'],
            'intent': 'aggregate'
        }
    )
    print(f"  CSV: {result_3['csv']}")
    print(f"  PNG: {result_3['png']}")
    print(f"  Query ID: {result_3['query_id']}\n")

    print("Session History:")
    print("-" * 80)
    for record in session.get_history_summary():
        print(f"  {record['query_id']} ({record['timestamp']})")
        print(f"    Question: {record['question']}")
        print(f"    Rows: {record['result_count']}")
        print(f"    Engine: {record['engine_mode']}")
        print()

    print("Session Statistics:")
    print("-" * 80)
    stats = session.get_stats()
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Total Rows: {stats['total_rows']}")
    print(f"  Tables Queried: {', '.join(stats.get('tables_queried', []))}")
    print(f"  Intents Used: {', '.join(stats.get('intents_used', []))}")
    print(f"  Created: {stats.get('created_at')}")
    print(f"  Last Query: {stats.get('last_query_at')}")

    print(f"\nSession data saved to: {session.user_dir}")
    print("Files:")
    for f in os.listdir(session.user_dir):
        fpath = os.path.join(session.user_dir, f)
        fsize = os.path.getsize(fpath)
        print(f"  {f} ({fsize:,} bytes)")
