# Scraped Concurrency Detector

**Type:** `scraped-concurrency-detector`

Computes pool saturation from engine telemetry instead of router-local accounting. Every router
replica in front of the same pool reads the same scraped state, so the signal needs no shared
counters between replicas.

```
saturation = (sum over pool endpoints of running + waiting  +  own recent dispatches)
             / (maxConcurrency * candidates with fresh metrics)
```

- **Capacity** comes from the candidates of the stage being evaluated. Scoped to `decode` in a
  `max-saturation-detector`, that is the decode endpoints.
- **Load** comes from every endpoint in the pool (`flowcontrol.SaturationPoolFromContext`), so a
  request still in prefill counts against decode capacity, matching router-local accounting that
  counts a request from dispatch to end of stream.
- **Own recent dispatches** cover the interval between dispatch and the first engine scrape that
  includes the request. A peer replica's dispatches in that interval are not seen.
- **Stale telemetry**: a stale endpoint keeps contributing its last scraped load; a stale
  candidate contributes no capacity. With no fresh candidate the pool is reported saturated.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `maxConcurrency` | `100` | In-flight requests one candidate endpoint is sized to hold. |
| `metricsStalenessThreshold` | `2s` | Age after which an endpoint's metrics are stale. |
| `ownDispatchWindow` | `1s` | How long a dispatched request counts before the scrape is trusted to include it. |

## Example

```yaml
- type: scraped-concurrency-detector
  name: decode-scraped
  parameters: {maxConcurrency: 5}
- type: max-saturation-detector
  name: admission-split
  parameters:
    detectors: [decode-scraped, prefill-queue]
    stages: {decode-scraped: [decode], prefill-queue: [prefill]}
```

Alpha: requires `--allow-experimental-plugins`.
