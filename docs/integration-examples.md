# Integration Examples

This guide shows how to integrate CoreVital into your application using the Library API (`CoreVitalMonitor`).

## Flask Example

```python
from flask import Flask, request, jsonify
from CoreVital import CoreVitalMonitor

app = Flask(__name__)

# Initialize monitor (reuse across requests)
monitor = CoreVitalMonitor(
    capture_mode="summary",  # Use "on_risk" for production
    intervene_on_risk_above=0.8,
)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    model_id = data.get("model", "gpt2")
    max_tokens = data.get("max_tokens", 50)
    
    # Run instrumented generation
    try:
        monitor.run(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_tokens,
            device="auto",
        )
        
        # Get results
        summary = monitor.get_summary()
        risk_score = monitor.get_risk_score()
        health_flags = monitor.get_health_flags()
        
        # Check if intervention needed
        if monitor.should_intervene():
            return jsonify({
                "error": "Model health check failed",
                "risk_score": risk_score,
                "health_flags": health_flags,
            }), 503
        
        # Return successful response
        return jsonify({
            "output": summary.get("output_text"),
            "risk_score": risk_score,
            "health_flags": health_flags,
            "trace_id": summary.get("trace_id"),
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
```

## FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from CoreVital import CoreVitalMonitor

app = FastAPI(title="LLM API with CoreVital")

# Initialize monitor
monitor = CoreVitalMonitor(
    capture_mode="on_risk",  # Summary by default, full on risk
    intervene_on_risk_above=0.7,
)

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "gpt2"
    max_tokens: int = 50
    temperature: float = 0.8

class GenerateResponse(BaseModel):
    output: str
    risk_score: float
    health_flags: dict
    trace_id: str | None

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text with health monitoring."""
    try:
        monitor.run(
            model_id=request.model,
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        summary = monitor.get_summary()
        risk_score = monitor.get_risk_score()
        
        if monitor.should_intervene():
            raise HTTPException(
                status_code=503,
                detail=f"Model health check failed: risk_score={risk_score}",
            )
        
        return GenerateResponse(
            output=summary.get("output_text", ""),
            risk_score=risk_score,
            health_flags=monitor.get_health_flags(),
            trace_id=summary.get("trace_id"),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

## Sampling Strategy (Production)

For high-volume production, instrument only a subset of requests:

```python
import random
from CoreVital import CoreVitalMonitor

monitor = CoreVitalMonitor(capture_mode="summary")
SAMPLING_RATE = 0.01  # 1% of requests

def generate_with_sampling(prompt: str, model_id: str):
    """Generate with probabilistic sampling."""
    # Always generate
    output = generate_without_monitoring(prompt, model_id)
    
    # Sample for monitoring
    if random.random() < SAMPLING_RATE:
        monitor.run(model_id, prompt, max_new_tokens=50)
        
        # Log metrics or alert if needed
        if monitor.should_intervene():
            logger.warning(
                f"High risk detected: {monitor.get_risk_score()}",
                extra={"trace_id": monitor.get_summary().get("trace_id")},
            )
    
    return output
```

## Async/Background Monitoring

Run monitoring in the background to avoid blocking:

```python
import asyncio
from CoreVital import CoreVitalMonitor

monitor = CoreVitalMonitor(capture_mode="summary")

async def monitor_async(prompt: str, model_id: str):
    """Run monitoring asynchronously."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: monitor.run(model_id, prompt, max_new_tokens=50),
    )
    return monitor.get_summary()

# Usage
async def generate_with_async_monitoring(prompt: str):
    output = generate_without_monitoring(prompt)
    
    # Monitor in background (fire and forget)
    asyncio.create_task(monitor_async(prompt, "gpt2"))
    
    return output
```

## Error Handling & Retry

Use CoreVital's health signals to decide when to retry:

```python
from CoreVital import CoreVitalMonitor

monitor = CoreVitalMonitor(
    capture_mode="summary",
    intervene_on_risk_above=0.8,
)

def generate_with_retry(prompt: str, model_id: str, max_retries: int = 3):
    """Generate with retry on health failure."""
    for attempt in range(max_retries):
        monitor.run(model_id, prompt, max_new_tokens=50)
        
        if not monitor.should_intervene():
            return monitor.get_summary().get("output_text")
        
        logger.warning(
            f"Attempt {attempt + 1} failed: risk={monitor.get_risk_score()}"
        )
        
        # Optional: adjust temperature or other params
        # monitor.run(..., temperature=0.5)
    
    raise Exception("Failed after retries")
```

## Exporting to Monitoring Systems

Send CoreVital metrics to your existing monitoring stack:

```python
from CoreVital import CoreVitalMonitor
from prometheus_client import Counter, Gauge

# Prometheus metrics
risk_score_gauge = Gauge("corevital_risk_score", "Risk score per run")
health_flags_counter = Counter("corevital_health_flags", "Health flag occurrences", ["flag"])

monitor = CoreVitalMonitor(capture_mode="summary")

def generate_with_metrics(prompt: str, model_id: str):
    monitor.run(model_id, prompt, max_new_tokens=50)
    
    # Export to Prometheus
    risk_score_gauge.set(monitor.get_risk_score())
    health_flags = monitor.get_health_flags()
    for flag, value in health_flags.items():
        if value:
            health_flags_counter.labels(flag=flag).inc()
    
    return monitor.get_summary()
```

## See Also

- [Production Deployment Guide](production-deployment.md) — Sampling, database setup, alerting
- [Library API Reference](../src/CoreVital/monitor.py) — Full `CoreVitalMonitor` API
- [Dashboard](../dashboard.py) — Visualize reports
