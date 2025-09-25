# AI Trading System API Documentation

This document provides comprehensive documentation for the AI Trading System REST API.

## Overview

The AI Trading System API exposes the functionality of a multi-agent trading system through RESTful endpoints. The system uses specialized agents (Technical, Sentiment, Risk, and Portfolio) to analyze market data and generate trading signals.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication for development purposes. In production, consider implementing API key authentication.

## Endpoints

### Trade Signals

#### GET /signals/{symbol}

Generates trade signals for a given stock symbol by running the complete multi-agent analysis workflow.

**Parameters:**
- `symbol` (path): Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- `days` (query, optional): Number of historical days to analyze (default: 30, min: 1, max: 365)

**Response:**
```json
{
  "symbol": "AAPL",
  "analysis_period": {
    "start_date": "2025-08-26T00:00:00",
    "end_date": "2025-09-25T00:00:00",
    "days": 30
  },
  "final_decision": {
    "signal": "BUY",
    "confidence": 0.85,
    "reasoning": "Technical indicators show strong upward momentum...",
    "timestamp": "2025-09-25T07:30:00.123456"
  },
  "agent_decisions": {
    "technical": {
      "signal": "BUY",
      "confidence": 0.9,
      "reasoning": "RSI indicates oversold conditions...",
      "timestamp": "2025-09-25T07:29:58.654321"
    },
    "sentiment": {
      "signal": "HOLD",
      "confidence": 0.6,
      "reasoning": "Neutral sentiment from news analysis",
      "timestamp": "2025-09-25T07:29:59.123456"
    },
    "risk": {
      "signal": "APPROVE",
      "confidence": 1.0,
      "reasoning": "Risk metrics within acceptable limits",
      "timestamp": "2025-09-25T07:29:59.567890"
    }
  }
}
```

**Example Request:**
```bash
curl "http://localhost:8000/api/v1/signals/AAPL?days=30"
```

### System Status

#### GET /status

Returns the current system status including portfolio state and agent availability.

**Response:**
```json
{
  "status": "operational",
  "timestamp": "2025-09-25T07:30:00.123456",
  "portfolio": {
    "cash": 95000.0,
    "positions": {
      "AAPL": 50,
      "MSFT": 25
    }
  },
  "agents": {
    "technical": "available",
    "sentiment": "available (mock)",
    "risk": "available (mock)",
    "portfolio": "available (mock)"
  },
  "data_sources": {
    "yfinance": "available",
    "alpha_vantage": "available"
  },
  "caching": {
    "redis": "available"
  }
}
```

#### GET /health

Basic health check endpoint for load balancers and monitoring systems.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-25T07:30:00.123456",
  "service": "AI Trading System API"
}
```

### Monitoring

#### GET /metrics

Returns application performance metrics.

**Response:**
```json
{
  "timestamp": "2025-09-25T07:30:00.123456",
  "uptime_seconds": 3600.5,
  "requests_total": 42,
  "errors_total": 2,
  "average_request_duration_seconds": 0.234,
  "error_rate": 0.0476
}
```

#### GET /ping

Simple ping endpoint for load balancer health checks.

**Response:**
```json
{
  "status": "pong",
  "timestamp": "2025-09-25T07:30:00.123456"
}
```

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error description"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Data Models

### AgentDecision
```json
{
  "agent_name": "technical",
  "symbol": "AAPL",
  "signal": "BUY|SELL|HOLD",
  "confidence": 0.85,
  "reasoning": "Analysis explanation...",
  "supporting_data": {},
  "timestamp": "2025-09-25T07:30:00.123456"
}
```

### MarketData
```json
{
  "symbol": "AAPL",
  "price": 150.25,
  "volume": 45238900,
  "timestamp": "2025-09-25T07:30:00.123456",
  "ohlc": {
    "open": 149.50,
    "high": 151.00,
    "low": 148.75,
    "close": 150.25
  },
  "technical_indicators": {
    "RSI": 65.5,
    "MACD": 1.25
  }
}
```

## Rate Limiting

Currently, no rate limiting is implemented. Consider adding rate limiting in production.

## Development

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation where you can test all endpoints directly in your browser.

### Alternative Documentation

Visit `http://localhost:8000/redoc` for ReDoc format documentation.

## Deployment

The API is containerized and can be deployed using Docker:

```bash
# Using Docker Compose (includes Redis and PostgreSQL)
docker-compose up -d

# Using deployment script
./scripts/deploy.sh
```

## Environment Variables

- `OPENROUTER_API_KEY`: Required for LLM functionality
- `REDIS_URL`: Redis connection URL (default: redis://redis:6379)
- `DATABASE_URL`: PostgreSQL connection URL

## Notes

- The Sentiment, Risk, and Portfolio agents currently return mock decisions for development purposes
- Real agent implementations will be added in Phase 5
- The system uses OpenRouter for flexible LLM provider support
- All timestamps are in ISO 8601 format with UTC timezone