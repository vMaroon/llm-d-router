# Concurrency Detector Plugin

**Type:** `concurrency-detector`

Synchronous saturation detection and scheduling filter mechanism based on active in-flight request accounting.

## What it does

This plugin uses a two-tier approach to manage average pool load and protect individual endpoints.

### Role in Flow Control (The Gatekeeper)
The detector implements the `SaturationDetector` interface to provide a utilization gradient, allowing the Flow Controller to apply proportional backpressure.

    PoolSaturation = Aggregate Inflight Load / Aggregate Pool Capacity

In token mode, the numerator includes published endpoint tokens and pending input-token dispatch reservations. The denominator is the sum of endpoint token capacities. Before tokenization, pending input uses one estimated token per raw request byte; this is not an exact context length.

Hybrid mode takes the maximum of three signals: the mean of endpoint-wise maximum request/token ratios, aggregate request pressure including pending dispatches, and aggregate token pressure including pending input. Reservations are released when request preparation publishes endpoint load or the dispatch ends. The stage-aware flow controller evaluates prefill and decode separately and combines their pressure with `max`.

**Heterogeneous Deployments:** Because this detector calculates saturation globally as a single aggregate fraction (in requests and tokens mode), it utilizes an aggregate queueing model. In deployments with heterogeneous compute (e.g., mixing H100 and L4 nodes), this heavily biases the pool saturation metric toward the state of the larger nodes. Contrast this with the Utilization Detector, which evaluates saturation as an unweighted average of individual endpoint scores.

### Role in Scheduling (The Traffic Shaper)
The detector implements the `Filter` interface to protect individual endpoints. In request mode it removes endpoints whose local in-flight request count reaches the safety limit. In token and hybrid modes it includes a tokenized incoming request's endpoint-specific uncached cost and removes endpoints whose projected load would exceed the safety limit. When tokenization is absent, the estimated incoming size is not used for projected rejection; existing endpoint load limits remain:

    EndpointLimit = Capacity * (1 + Headroom)
    ProjectedTokens = InflightTokens + IncomingUncachedTokens

This approach allows the Flow Controller to manage average pool load, while the Scheduler retains the flexibility to burst above ideal targets (the "Headroom") to satisfy affinity or scoring objectives.

## Inputs consumed

The detector reads endpoint request/token attributes from its configured in-flight load producer. That producer owns the request/response lifecycle. The detector separately tracks pending dispatch reservations before those attributes are published.

## Configuration

The plugin accepts JSON parameters decoding to the following fields:

- `concurrencyMode` (`string`): Evaluation mode. Valid values are `"requests"`, `"tokens"`, or `"hybrid"`. In `"hybrid"` mode both request and token accounting are evaluated. Pool saturation is computed per endpoint as the larger of that endpoint's request and token ratios, then averaged across endpoints, so an endpoint saturated on either dimension is reflected even when distinct endpoints saturate on different dimensions. An endpoint is filtered out when either its request load or its token load reaches the limit. (Default: `"requests"`)
- `maxConcurrency` (`int64`): Maximum requests in flight. Serves as the "ideal" request capacity for a single endpoint. Must be > 0. (Default: `100`)
- `maxTokenConcurrency` (`int64`): Maximum tokens in flight. The "tokens" mode equivalent of `maxConcurrency`. Must be > 0. (Default: `1000000`)
- `maxTokenConcurrencyByRole` (`map[string]int64`): Optional token-capacity overrides keyed by exact `llm-d.ai/role` values. Recognized roles are `prefill`, `decode`, `encode`, `encode-prefill`, `prefill-decode`, `encode-prefill-decode`, and the legacy `both`. Values must be positive. Unlisted or unlabeled endpoints use `maxTokenConcurrency`. Overrides apply to saturation and filtering, without changing token accounting. This permits a prefill work budget distinct from decode context capacity; the prefill value requires workload measurement, not derivation from KV size alone.
- `headroom` (`float64`): Allowed burst capacity above the ideal threshold, expressed as a fraction (e.g., `0.2` for 20%). Must be >= 0.0. (Default: `0.0`)
- `decodeSafety`: Optional utilization-detector configuration. Fresh decoder queue/KV telemetry is combined with logical saturation using `max`. Missing or stale telemetry contributes no pressure; logical accounting remains active.

## Trade-offs

Logical tokens are not physical unique KV pages. Shared prefixes, inaccurate estimates and output growth can separate logical pressure from engine occupancy. Streaming accounting depends on usage observations from the engine; optional telemetry is delayed and may be absent. Aggregate admission does not guarantee that every endpoint avoids a hotspot or preemption. Conservative byte reservations can also delay a cold burst more than exact token counts would.
