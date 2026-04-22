import os
import re
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger('gpdm.local_llm')


@dataclass
class ModelSpec:
    name: str
    family: str
    param_count: str
    min_ram_gb: float
    quantized: bool
    format: str
    description: str
    search_paths: List[str] = field(default_factory=list)
    hf_model_id: str = ''


MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name='llama-3-8b-instruct',
        family='llama3',
        param_count='8B',
        min_ram_gb=6.0,
        quantized=True,
        format='gguf',
        description='Meta Llama 3 8B Instruct (Q4_K_M quantized) — best quality for NLU',
        search_paths=[
            '~/.cache/llama/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
            '~/models/llama-3-8b-instruct.gguf',
            './models/llama-3-8b-instruct.gguf',
            '/opt/models/llama-3-8b-instruct.gguf',
        ],
        hf_model_id='QuantFactory/Meta-Llama-3-8B-Instruct-GGUF',
    ),
    ModelSpec(
        name='phi-3-mini-instruct',
        family='phi3',
        param_count='3.8B',
        min_ram_gb=3.0,
        quantized=True,
        format='gguf',
        description='Microsoft Phi-3 Mini Instruct (Q4_K_M) — fast, good quality',
        search_paths=[
            '~/.cache/phi/Phi-3-mini-4k-instruct-q4.gguf',
            '~/models/phi-3-mini-instruct.gguf',
            './models/phi-3-mini-instruct.gguf',
            '/opt/models/phi-3-mini-instruct.gguf',
        ],
        hf_model_id='microsoft/Phi-3-mini-4k-instruct-gguf',
    ),
    ModelSpec(
        name='tinyllama-1.1b-chat',
        family='tinyllama',
        param_count='1.1B',
        min_ram_gb=1.0,
        quantized=True,
        format='gguf',
        description='TinyLlama 1.1B Chat (Q4_K_M) — minimal RAM, acceptable quality',
        search_paths=[
            '~/.cache/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
            '~/models/tinyllama-1.1b-chat.gguf',
            './models/tinyllama-1.1b-chat.gguf',
            '/opt/models/tinyllama-1.1b-chat.gguf',
        ],
        hf_model_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    ),
]


class LocalModelLoader:

    def __init__(self, model_dir: str = None, allow_download: bool = False):
        self.model_dir = model_dir or os.path.expanduser('~/models')
        self.allow_download = allow_download
        self._model = None
        self._model_spec = None
        self._backend = None

    def detect_available(self) -> List[Tuple[ModelSpec, str]]:
        available = []
        for spec in MODEL_REGISTRY:
            paths = list(spec.search_paths)
            if self.model_dir:
                paths.append(os.path.join(self.model_dir, f'{spec.name}.gguf'))
                paths.append(os.path.join(self.model_dir, spec.name))

            for path in paths:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    available.append((spec, expanded))
                    logger.info("Found model: %s at %s", spec.name, expanded)
                    break

        return available

    def detect_backends(self) -> Dict[str, bool]:
        backends = {}

        try:
            import llama_cpp
            backends['llama_cpp'] = True
            logger.info("Backend available: llama-cpp-python")
        except ImportError:
            backends['llama_cpp'] = False

        try:
            import ctransformers
            backends['ctransformers'] = True
            logger.info("Backend available: ctransformers")
        except ImportError:
            backends['ctransformers'] = False

        try:
            import transformers
            backends['transformers'] = True
            logger.info("Backend available: transformers")
        except ImportError:
            backends['transformers'] = False

        return backends

    def load(self, preferred_model: str = None) -> bool:
        backends = self.detect_backends()
        available = self.detect_available()

        if not available:
            logger.warning("No local LLM models found. LLM features disabled.")
            return False

        if not any(backends.values()):
            logger.warning("No LLM inference backends installed. "
                          "Install llama-cpp-python: pip install llama-cpp-python")
            return False

        selected = None
        if preferred_model:
            for spec, path in available:
                if spec.name == preferred_model:
                    selected = (spec, path)
                    break

        if not selected:
            selected = available[0]

        spec, path = selected
        logger.info("Loading model: %s from %s", spec.name, path)

        try:
            if spec.format == 'gguf' and backends.get('llama_cpp'):
                self._load_llama_cpp(path, spec)
            elif spec.format == 'gguf' and backends.get('ctransformers'):
                self._load_ctransformers(path, spec)
            elif backends.get('transformers'):
                self._load_transformers(path, spec)
            else:
                logger.error("No compatible backend for model format: %s", spec.format)
                return False

            self._model_spec = spec
            logger.info("Model loaded successfully: %s (%s)", spec.name, self._backend)
            return True

        except Exception as e:
            logger.error("Failed to load model %s: %s", spec.name, e)
            return False

    def _load_llama_cpp(self, path: str, spec: ModelSpec):
        from llama_cpp import Llama
        self._model = Llama(
            model_path=path,
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        self._backend = 'llama_cpp'

    def _load_ctransformers(self, path: str, spec: ModelSpec):
        from ctransformers import AutoModelForCausalLM
        model_type = {
            'llama3': 'llama',
            'phi3': 'phi',
            'tinyllama': 'llama',
            'mistral': 'mistral',
        }.get(spec.family, 'llama')

        self._model = AutoModelForCausalLM.from_pretrained(
            path,
            model_type=model_type,
            context_length=4096,
        )
        self._backend = 'ctransformers'

    def _load_transformers(self, path: str, spec: ModelSpec):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map='cpu',
            torch_dtype='auto',
        )
        self._model = pipeline('text-generation', model=model, tokenizer=tokenizer)
        self._backend = 'transformers'

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        if not self._model:
            raise RuntimeError("No model loaded. Call load() first.")

        try:
            if self._backend == 'llama_cpp':
                result = self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=['\n\n', '</s>', '<|eot_id|>'],
                    echo=False,
                )
                return result['choices'][0]['text'].strip()

            elif self._backend == 'ctransformers':
                return self._model(prompt, max_new_tokens=max_tokens, temperature=temperature)

            elif self._backend == 'transformers':
                result = self._model(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
                return result[0]['generated_text'][len(prompt):].strip()

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return ''

    @property
    def model_name(self) -> str:
        return self._model_spec.name if self._model_spec else 'none'

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class PromptBuilder:

    def __init__(self, schema_graph):
        self.graph = schema_graph
        self._schema_description = self._build_schema_description()

    def _build_schema_description(self) -> str:
        parts = []
        parts.append("DATABASE SCHEMA:")
        for table_name, concept in self.graph.tables.items():
            cols = list(self.graph.columns.get(table_name, {}).keys())
            col_str = ', '.join(cols[:15])
            if len(cols) > 15:
                col_str += f' ... ({len(cols)} total)'
            parts.append(f"  {table_name} ({concept.description}): {col_str}")

        parts.append("\nKEY RELATIONSHIPS:")
        parts.append("  members.MEMBER_ID → claims, encounters, prescriptions, diagnoses, referrals, appointments")
        parts.append("  claims.RENDERING_NPI → providers.NPI")
        parts.append("  claims.ENCOUNTER_ID → encounters.ENCOUNTER_ID")
        parts.append("  encounters.RENDERING_NPI → providers.NPI")
        parts.append("  prescriptions.PRESCRIBING_NPI → providers.NPI")

        parts.append("\nMONEY COLUMNS: claims.PAID_AMOUNT, claims.BILLED_AMOUNT, claims.ALLOWED_AMOUNT, prescriptions.COST")
        parts.append("MEASURE COLUMNS: encounters.LENGTH_OF_STAY, members.RISK_SCORE, providers.PANEL_SIZE")
        return '\n'.join(parts)

    def build_intent_prompt(self, question: str) -> str:
        return f"""You are a healthcare analytics SQL assistant. Given a natural language question about a healthcare database, extract the structured intent.

{self._schema_description}

TASK: Parse the following question into structured intent. Return ONLY valid JSON.

INTENT TYPES: count, aggregate, compare, trend, rank, rate, list, summary, correlate, exists
AGG FUNCTIONS: COUNT, SUM, AVG, MAX, MIN
FILTER OPERATORS: =, >, <, >=, <=, LIKE, IN, BETWEEN, !=

Question: "{question}"

Return JSON:
{{"intent": "<type>", "sub_intent": "<avg|sum|max|min|count|percentage>", "tables": ["<table1>", ...], "agg_function": "<AGG>", "agg_column": "<COLUMN>", "agg_table": "<table>", "group_by": [["<table>", "<column>"]], "filters": [{{"column": "<COL>", "operator": "<OP>", "value": "<VAL>", "table": "<table>"}}], "order_by": "<asc|desc|>", "limit": <N|null>, "temporal": <true|false>, "time_granularity": "<month|quarter|year|day|>", "comparison": <true|false>, "compare_values": ["<val1>", "<val2>"]}}

JSON:"""

    def build_sql_prompt(self, question: str, intent_json: str) -> str:
        return f"""You are a healthcare analytics SQL expert. Generate a single SQLite SQL query for the question.

{self._schema_description}

RULES:
- Use SQLite syntax (SUBSTR not SUBSTRING, no LIMIT with OFFSET unless needed)
- Always use table prefixes in JOINs (e.g., claims.MEMBER_ID)
- For money: SUM(CAST(column AS REAL))
- For counts of entities: COUNT(DISTINCT id_column) when JOINed
- For "which X has most Y": GROUP BY X, ORDER BY COUNT(*) DESC LIMIT 1
- Never use SELECT * in production queries

Parsed intent: {intent_json}

Question: "{question}"

SQL:"""


class LLMIntentParser:

    def __init__(self, schema_graph, model_loader: LocalModelLoader = None,
                 fallback_parser=None):
        self.graph = schema_graph
        self.model = model_loader
        self.fallback = fallback_parser
        self.prompt_builder = PromptBuilder(schema_graph)
        self._parse_count = 0
        self._fallback_count = 0

    @property
    def llm_available(self) -> bool:
        return self.model is not None and self.model.is_loaded

    def parse(self, question: str):
        from intent_parser import ParsedIntent, ParsedFilter, normalize_typos

        self._parse_count += 1

        if not self.llm_available:
            return self._fallback_parse(question)

        try:
            prompt = self.prompt_builder.build_intent_prompt(question)

            t0 = time.time()
            response = self.model.generate(prompt, max_tokens=512, temperature=0.1)
            elapsed = time.time() - t0
            logger.info("LLM intent parse: %.2fs for '%s'", elapsed, question[:50])

            intent_data = self._parse_json_response(response)
            if not intent_data:
                logger.warning("LLM returned malformed JSON, falling back to rules")
                return self._fallback_parse(question)

            normalized = normalize_typos(question.lower().strip())
            intent = ParsedIntent(
                original_question=question,
                normalized_question=normalized,
                intent=intent_data.get('intent', 'count'),
            )

            intent.sub_intent = intent_data.get('sub_intent', '')
            intent.tables = intent_data.get('tables', [])
            intent.agg_function = intent_data.get('agg_function', '')
            intent.agg_column = intent_data.get('agg_column', '')
            intent.agg_table = intent_data.get('agg_table', '')
            intent.order_by = intent_data.get('order_by', '')
            intent.limit = intent_data.get('limit')
            intent.temporal = intent_data.get('temporal', False)
            intent.time_granularity = intent_data.get('time_granularity', '')
            intent.comparison = intent_data.get('comparison', False)
            intent.compare_values = intent_data.get('compare_values', [])

            for gb in intent_data.get('group_by', []):
                if isinstance(gb, list) and len(gb) == 2:
                    intent.group_by.append((gb[0], gb[1]))
                elif isinstance(gb, str):
                    tables_with_col = self.graph.column_to_tables.get(gb, [])
                    if tables_with_col:
                        intent.group_by.append((tables_with_col[0], gb))

            for f in intent_data.get('filters', []):
                if isinstance(f, dict) and 'column' in f:
                    intent.filters.append(ParsedFilter(
                        column=f['column'],
                        operator=f.get('operator', '='),
                        value=f.get('value', ''),
                        table_hint=f.get('table', ''),
                        confidence=0.85,
                    ))

            words = re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', normalized)
            col_matches = self.graph.find_columns_for_words(
                words, intent.tables or None, raw_question=normalized
            )
            intent.columns = [(t, c) for t, c, st, conf in col_matches[:10]]

            if not intent.tables:
                logger.warning("LLM returned no tables, falling back to rules")
                return self._fallback_parse(question)

            valid_tables = [t for t in intent.tables if t in self.graph.tables]
            if not valid_tables:
                logger.warning("LLM returned invalid table names: %s", intent.tables)
                return self._fallback_parse(question)
            intent.tables = valid_tables

            intent.confidence = 0.85
            return intent

        except Exception as e:
            logger.error("LLM parse failed: %s, falling back to rules", e)
            return self._fallback_parse(question)

    def _fallback_parse(self, question: str):
        self._fallback_count += 1
        if self.fallback:
            return self.fallback.parse(question)
        from intent_parser import IntentParser
        parser = IntentParser(self.graph)
        return parser.parse(question)

    def _parse_json_response(self, response: str) -> Optional[Dict]:
        if not response:
            return None

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def stats(self) -> Dict[str, Any]:
        return {
            'total_parses': self._parse_count,
            'fallback_count': self._fallback_count,
            'llm_available': self.llm_available,
            'model_name': self.model.model_name if self.model else 'none',
            'llm_rate': (self._parse_count - self._fallback_count) / max(self._parse_count, 1),
        }


class ConversationContext:

    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.last_tables: List[str] = []
        self.last_filters: List[Any] = []
        self.last_intent: str = ''
        self.last_agg_column: str = ''

    def add_turn(self, question: str, intent, sql: str = '', result: Any = None):
        self.history.append({
            'question': question,
            'intent': intent.intent if hasattr(intent, 'intent') else str(intent),
            'tables': intent.tables if hasattr(intent, 'tables') else [],
            'sql': sql,
            'timestamp': time.time(),
        })

        if hasattr(intent, 'tables') and intent.tables:
            self.last_tables = intent.tables
        if hasattr(intent, 'filters'):
            self.last_filters = intent.filters
        if hasattr(intent, 'intent'):
            self.last_intent = intent.intent
        if hasattr(intent, 'agg_column') and intent.agg_column:
            self.last_agg_column = intent.agg_column

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def is_followup(self, question: str) -> bool:
        if not self.history:
            return False

        q_lower = question.lower().strip()
        followup_patterns = [
            r'^(?:and|but|also|what about|how about|now|ok)\b',
            r'^(?:show|break|split|group)\s+(?:it|that|this|them)\b',
            r'^by\s+\w+',
            r'^for\s+\w+',
            r'^(?:same|those|these)\b',
            r'^(?:and|or)\s+by\b',
            r'^what\s+(?:if|about)\b',
        ]
        return any(re.search(p, q_lower) for p in followup_patterns)

    def enrich_followup(self, question: str, intent) -> None:
        if not self.history:
            return

        if not intent.tables and self.last_tables:
            intent.tables = list(self.last_tables)

        if not intent.filters and self.last_filters:
            intent.filters = list(self.last_filters)

        if not intent.agg_column and self.last_agg_column:
            intent.agg_column = self.last_agg_column

    def get_context_summary(self) -> str:
        if not self.history:
            return ''

        recent = self.history[-3:]
        parts = ["CONVERSATION CONTEXT:"]
        for turn in recent:
            parts.append(f"  Q: {turn['question']}")
            parts.append(f"  → intent={turn['intent']}, tables={turn['tables']}")
        return '\n'.join(parts)


class LLMConfig:

    def __init__(self, config_dict: Dict = None):
        cfg = config_dict or {}
        self.enabled = cfg.get('LLM_ENABLED', 'false').lower() in ('true', '1', 'yes')
        self.model_dir = cfg.get('LLM_MODEL_DIR', os.path.expanduser('~/models'))
        self.preferred_model = cfg.get('LLM_MODEL', '')
        self.allow_download = cfg.get('LLM_ALLOW_DOWNLOAD', 'false').lower() in ('true', '1', 'yes')
        self.max_tokens = int(cfg.get('LLM_MAX_TOKENS', '512'))
        self.temperature = float(cfg.get('LLM_TEMPERATURE', '0.1'))
        self.fallback_to_rules = cfg.get('LLM_FALLBACK', 'true').lower() in ('true', '1', 'yes')
        self.context_enabled = cfg.get('LLM_CONTEXT', 'true').lower() in ('true', '1', 'yes')


def create_intent_parser(schema_graph, config: Dict = None, interactive: bool = False):
    from intent_parser import IntentParser

    llm_config = LLMConfig(config)

    if interactive and not llm_config.enabled:
        try:
            response = input(
                "\nEnable local LLM for enhanced language understanding? "
                "(requires ~6GB RAM) [y/N]: "
            ).strip().lower()
            if response in ('y', 'yes'):
                llm_config.enabled = True
                print("✓ LLM enabled — scanning for local models...")
        except (EOFError, KeyboardInterrupt):
            pass

    rule_parser = IntentParser(schema_graph)

    if not llm_config.enabled:
        logger.debug("External LLM not configured — using built-in intelligence engines")
        return rule_parser

    loader = LocalModelLoader(
        model_dir=llm_config.model_dir,
        allow_download=llm_config.allow_download,
    )

    available = loader.detect_available()
    if not available:
        logger.warning("LLM enabled but no models found in %s", llm_config.model_dir)
        logger.warning("To use LLM, place a GGUF model file in: %s", llm_config.model_dir)
        logger.warning("Supported: Llama 3 8B, Phi-3 Mini, TinyLlama 1.1B")
        logger.warning("Falling back to rule-based parser")
        return rule_parser

    backends = loader.detect_backends()
    if not any(backends.values()):
        logger.warning("LLM enabled but no inference backend installed.")
        logger.warning("Install: pip install llama-cpp-python --break-system-packages")
        logger.warning("Falling back to rule-based parser")
        return rule_parser

    success = loader.load(preferred_model=llm_config.preferred_model)
    if not success:
        logger.warning("Failed to load LLM model. Falling back to rule-based parser")
        return rule_parser

    llm_parser = LLMIntentParser(
        schema_graph=schema_graph,
        model_loader=loader,
        fallback_parser=rule_parser if llm_config.fallback_to_rules else None,
    )

    logger.info("LLM intent parser active: %s (with rule-based fallback)",
                loader.model_name)
    return llm_parser


def diagnose_llm_setup() -> Dict[str, Any]:
    report = {
        'backends': {},
        'models_found': [],
        'models_not_found': [],
        'recommendation': '',
    }

    loader = LocalModelLoader()
    report['backends'] = loader.detect_backends()

    available = loader.detect_available()
    report['models_found'] = [
        {'name': spec.name, 'path': path, 'params': spec.param_count}
        for spec, path in available
    ]
    report['models_not_found'] = [
        spec.name for spec in MODEL_REGISTRY
        if spec.name not in {s.name for s, p in available}
    ]

    if not any(report['backends'].values()):
        report['recommendation'] = (
            "Install an inference backend:\n"
            "  pip install llama-cpp-python --break-system-packages\n"
            "Then download a model:\n"
            "  Place a GGUF file in ~/models/"
        )
    elif not available:
        report['recommendation'] = (
            "Backend ready but no models found.\n"
            "Download a quantized model (Q4_K_M recommended):\n"
            "  Llama 3 8B: ~4.5GB, best quality\n"
            "  Phi-3 Mini: ~2.3GB, good quality, faster\n"
            "  TinyLlama: ~0.7GB, minimal quality\n"
            "Place the .gguf file in ~/models/"
        )
    else:
        best = available[0][0]
        report['recommendation'] = f"Ready! Best model: {best.name} ({best.param_count})"

    return report


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  LOCAL LLM ENGINE — Diagnostic Report")
    print("=" * 60)

    report = diagnose_llm_setup()

    print("\nInference Backends:")
    for backend, available in report['backends'].items():
        status = "✓ installed" if available else "✗ not found"
        print(f"  {backend}: {status}")

    print("\nLocal Models:")
    if report['models_found']:
        for m in report['models_found']:
            print(f"  ✓ {m['name']} ({m['params']}) — {m['path']}")
    else:
        print("  No models found")

    if report['models_not_found']:
        print("\n  Missing models:")
        for name in report['models_not_found']:
            print(f"    ✗ {name}")

    print(f"\nRecommendation:\n  {report['recommendation']}")

    if report['models_found'] and any(report['backends'].values()):
        print("\n" + "=" * 60)
        print("  Running quick inference test...")
        print("=" * 60)

        loader = LocalModelLoader()
        if loader.load():
            test_prompt = "Parse this question: 'how many claims are denied?' Return JSON with intent and tables."
            result = loader.generate(test_prompt, max_tokens=200)
            print(f"\n  Prompt: {test_prompt[:60]}...")
            print(f"  Response: {result[:200]}")
            print(f"  Model: {loader.model_name}")
            print(f"  Backend: {loader._backend}")
        print("\n✓ Test complete")
