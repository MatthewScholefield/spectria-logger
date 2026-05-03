# keras-spectra

Keras callback + HTTP/SSE server for live training visualization in [Spectra](https://github.com/MatthewScholefield/spectra).

A lightweight alternative to TensorBoard with a cleaner, more intentional visualization experience.

## Usage

### In your training script — log metrics

```bash
pip install keras-spectra
```

```python
from keras_spectra import SpectraCallback, as_keras_callback

model.fit(x, y, callbacks=[
    as_keras_callback(SpectraCallback(
        project="mnist",
        run="baseline",
        config={"model": "resnet50", "lr": 0.01, "optimizer": "adam", "batch_size": 64},
    ))
])
```

Runs with a shared baseline are grouped in Spectra:

```python
as_keras_callback(SpectraCallback(
    project="mnist",
    run="lr-0.001",
    baseline="baseline",
    config={"model": "resnet50", "lr": 0.001, "optimizer": "adam", "batch_size": 64},
))
```

### Visualize — serve the logs

No install needed:

```bash
uvx --from "keras-spectra[server]" keras-spectra serve
```

Then open Spectra, click **Connect**, enter `http://localhost:8420`, and select your runs.

Options:

```
keras-spectra serve --logdir ./spectra_logs --port 8420 --host 127.0.0.1
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
# {"spectra_version":1,"project":"mnist","run":"lr-0.001","baseline":"baseline","config":{"lr":0.001},"created_at":1714732800}
{"epoch":0,"loss":0.6931,"accuracy":0.5123,"_ts":1714732800}
{"epoch":1,"loss":0.4201,"accuracy":0.8102,"_ts":1714732806}
```

## License

MIT
