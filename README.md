# spectria-python

JSONL logging library + HTTP/SSE server for live training visualization in [Spectria](https://github.com/MatthewScholefield/spectria).

A lightweight alternative to TensorBoard with a cleaner, more intentional visualization experience.

This repo publishes two packages:

| Package | Description |
|---|---|
| `spectria-logger` | Logging library (`spectria.logger`) — zero dependencies |
| `spectria` | Server + CLI — depends on `spectria-logger`, `starlette`, `uvicorn` |

## Usage

### In your training script — log metrics

```bash
pip install spectria-logger
```

```python
from spectria.logger import RunWriter

writer = RunWriter(project="mnist", run="baseline", config={"lr": 0.01})
writer.write_row({"epoch": 0, "loss": 0.69, "accuracy": 0.51})
```

### Keras integration

```bash
pip install spectria-logger[keras]
```

```python
from spectria.logger import SpectriaCallback, as_keras_callback

model.fit(x, y, callbacks=[
    as_keras_callback(SpectriaCallback(
        project="mnist",
        run="baseline",
        config={"model": "resnet50", "lr": 0.01, "optimizer": "adam", "batch_size": 64},
    ))
])
```

Runs with a shared baseline are grouped in Spectria:

```python
as_keras_callback(SpectriaCallback(
    project="mnist",
    run="lr-0.001",
    baseline="baseline",
    config={"model": "resnet50", "lr": 0.001, "optimizer": "adam", "batch_size": 64},
))
```

### Visualize — serve the logs

```bash
uvx spectria serve
```

Then open Spectria, click **Connect**, enter `http://localhost:8420`, and select your runs.

Options:

```
spectria serve --logdir ./spectria_logs --port 8420 --host 127.0.0.1
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/projects` | List all projects |
| `GET /api/projects/{name}/runs` | List runs with metadata |
| `GET /api/projects/{name}/runs/{run}/data` | Full JSON snapshot |
| `GET /api/projects/{name}/runs/{run}/events` | SSE stream for live data |

## Data Format

Training events are stored as JSONL:

```jsonl
# {"spectria_version":1,"project":"mnist","run":"lr-0.001","baseline":"baseline","config":{"lr":0.001},"created_at":1714732800}
{"epoch":0,"loss":0.6931,"accuracy":0.5123,"_ts":1714732800}
{"epoch":1,"loss":0.4201,"accuracy":0.8102,"_ts":1714732806}
```

## License

MIT
