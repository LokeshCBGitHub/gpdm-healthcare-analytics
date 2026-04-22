# GPDM Healthcare Analytics Platform

A comprehensive, fully on-premise healthcare analytics platform with natural language query processing, 16-domain interactive dashboards, 7-model statistical forecasting, and real-time data exploration.

## Architecture

- **Query Pipeline**: Natural language to SQL with schema-aware query construction
- **Analytics Engine**: Multi-query decomposition, statistical analysis, correlation detection
- **Dashboard System**: 16 healthcare domains with interactive charts and drill-downs
- **Forecasting**: 7-model ensemble (Holt-Winters, ARIMA, Exponential Smoothing, Linear Regression, Prophet-style, Bayesian Bootstrap, Monte Carlo)
- **Data Quality**: Referential integrity validation, cross-product detection, statistical significance testing

## Quick Start

```bash
cd scripts
python3 dashboard_server.py --db ../data/healthcare_71k.db --port 5000
```

Access at `https://localhost:5000`

## License

Proprietary. See [LICENSE](LICENSE) for full terms.

Copyright (c) 2024-2026 Ivan Holden. All Rights Reserved.
