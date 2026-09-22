# Utilization Detector Plugin

**Type:** `utilization-detector`

Reactive saturation detection and scheduling filter based on telemetry from LLM serving backends.

> [!NOTE]
> This plugin is enabled by default when flow control is enabled. You do not need to explicitly declare it in your configuration.

## What it does

This plugin uses a two-tier approach to manage average pool load and protect individual endpoints.

### Role in Flow Control (The Gatekeeper)
The detector implements the `SaturationDetector` interface to provide a utilization gradient, allowing the Flow Controller to apply proportional backpressure when the system is overloaded.

It relies on a "roofline model", evaluating both the queue depth and the KV cache utilization to find the most constrained resource for a given endpoint:

    EndpointScore = max(QueueDepth / QueueThreshold, KVCacheUsage / KVCacheThreshold)

The global pool saturation is then evaluated across all candidate endpoints as a gradient:

    PoolSaturation = Average(EndpointScore)

**Heterogeneous Deployments:** Because this detector calculates saturation as an unweighted average of individual endpoint scores, it treats all endpoints equally regardless of their physical capacity. In deployments with heterogeneous compute (e.g., mixing H100 and L4 nodes), a small, saturated endpoint has the exact same impact on global backpressure as a massive, saturated endpoint. Contrast this with the Concurrency Detector, which evaluates saturation as a single aggregate fraction, biasing toward larger endpoints.
Missing or stale metrics contribute zero telemetry-based pressure, while the endpoint remains in the pool denominator. This is fail-open behavior, not evidence that the endpoint is idle. A fleet-wide metrics outage does not halt admission by itself. An empty endpoint pool still reports saturation 1.0 because no serving capacity exists.

The `flow_control_stale_endpoints` gauge and rate-limited log identify telemetry loss. Fresh queue/KV measurements still contribute pressure. When used as the concurrency detector's `decodeSafety` floor, missing telemetry leaves logical request/token accounting in control. A standalone utilization detector has no such accounting fallback; during a full telemetry outage it cannot bound overload.

### Role in Scheduling (The Traffic Shaper)
The detector implements the `Filter` interface to protect individual endpoints. It removes endpoints from candidate lists when fresh telemetry exceeds specific safety limits. Missing or stale telemetry does not exclude an endpoint:

    MaxQueueLimit = QueueThreshold * (1 + Headroom)
    MaxKVCacheLimit = min(1.0, KVCacheThreshold * (1 + Headroom))

This approach allows the Flow Controller to manage average pool load, while the Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy affinity or scoring objectives.

**Fail-Open Fallback:** If all candidate endpoints exceed the safety limits, the filter returns the original list, allowing the scheduler's scorers to pick the least-bad option.

## Inputs consumed

The plugin consumes standard metrics from endpoints:
- `WaitingQueueSize` (Queue depth metric).
- `KVCacheUsagePercent` (KV cache utilization metric).
- `UpdateTime` (Timestamp used to calculate metric staleness).

## Configuration

The plugin accepts JSON parameters decoding to the following fields:

- `queueDepthThreshold` (`int`): Target waiting queue depth limit. Serves as the "ideal" queue capacity for a single endpoint. Must be > 0. (Default: `5`)
- `kvCacheUtilThreshold` (`float64`): Target KV cache memory utilization limit, expressed as a fraction. Must be in `(0.0, 1.0]`. (Default: `0.8`)
- `metricsStalenessThreshold` (`string` / duration): Maximum age of usable telemetry. Missing or stale metrics contribute no saturation pressure and do not exclude an endpoint. Must be > 0. (Default: `"200ms"`)
- `headroom` (`float64`): Allowed burst capacity above the ideal thresholds, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)

## Trade-offs

Unlike the Concurrency Detector, this approach operates as a closed-loop controller. It is immune to estimation divergence and reflects the actual performance limits of the continuous batching engine's memory manager. However, it suffers from two fundamental flaws of reactive systems:

1. **Telemetry Staleness (The Thundering Herd)**: Because it relies on asynchronous polling, the view of the endpoint state is perpetually delayed. A sudden burst of traffic can create a severe "thundering herd" condition, where the scheduler routes massive request volumes to a seemingly healthy endpoint before the next metric interval reveals it is completely saturated.
2. **Reactive Backpressure**: By definition, this detector only signals saturation after the inference engine is already under physical duress. It cannot preemptively shield an endpoint from an initial queue buildup; it can only throttle traffic after the physical limits have been breached and latency (TTFT/TPOT) has already degraded.
