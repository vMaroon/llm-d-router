# P2P Source Producer Plugin

**Type:** `p2p-source-producer`

Sets the `x-kv-cache-source-host-port` header to an endpoint within one block of the most cached prompt prefix, so the routing sidecar can pull those blocks over the P2P connector instead of recomputing them. Runs in the request handling's `DataProducer` phase before scheduling, then emits the header in `PreRequest` after the scheduling decision.

For each request the plugin consumes the per-endpoint `PrefixCacheMatchInfo` of a prefix-cache producer (`approx-prefix-cache-producer` or `precise-prefix-cache-producer`) and picks a source among the endpoints caching within one block of the most prompt tokens, weighted by `1/(1+waiting queue)` with a request-ID hash as the sampling coordinate; when the producer supplies per-tier data, only CPU-tier blocks count, since pulls are served from the source's CPU tier. Sampling within the one-block band, rather than argmax, keeps pull traffic from concentrating on a single replica of a widely-cached prefix. After scheduling, the header is set only when the chosen peer out-caches the computing pod by at least `minCachedTokenDelta` tokens; any inbound header value is removed.

**Parameters:**

- `prefixMatchInfoProducerName` (string, optional): Name of the prefix-cache producer instance to consume `PrefixCacheMatchInfo` from, e.g. `precise-prefix-cache-producer`. Empty selects the default (unnamed) producer.
- `minCachedTokenDelta` (int, optional, default: `1`): Minimum number of cached prompt tokens the best peer must hold beyond the computing pod for the header to be set. Must be `>= 1`. Higher values suppress pulls of short prefixes that are cheap to recompute.
- `prefillProfileName` (string, optional, default: `prefill`): Name of the P/D disaggregation prefill scheduling profile. The computing pod is read from this profile's target when present; otherwise the primary profile's target is used.
- `costModel` (object, optional): When set, the header is emitted only when the pull's gain exceeds its cost, evaluated after the `minCachedTokenDelta` floor. Omitted, the fixed-delta decision applies. See [Cost model](#cost-model).

**Configuration Example:**

```yaml
plugins:
  - type: precise-prefix-cache-producer
    parameters:
      tokenProcessorConfig:
        blockSize: 64
      kvEventsConfig:
        topicFilter: "kv@"
  - type: p2p-source-producer
    parameters:
      prefixMatchInfoProducerName: precise-prefix-cache-producer
      prefillProfileName: prefill
      minCachedTokenDelta: 1
```

## Cost model

A pull saves the computing pod the prefill of `delta` tokens (the source's cached-token advantage) and costs the transfer of those tokens, a fixed setup cost, and a re-queue delay when the computing pod has a waiting queue: the pulled request is re-admitted through that queue after its blocks arrive. On a loaded pod a recompute also delays every running request's decode steps, while a pull delays only the pulled request; `fleetWeight` credits the pull with that freed prefill. The header is emitted when

```text
delta * (prefill - transfer) + fleetWeight * running(d) * delta * prefill
  > transferFixed + busy(s) * sourceWait + busy(d) * requeue
```

where `running(d)` is the number of requests running on the computing pod and `busy(x)` is true when pod `x` has at least `busyQueueThreshold` requests waiting. At `fleetWeight: 0` the decision optimizes the pulled request's own latency only.

The cost model also changes how the source is chosen. Instead of sampling within one block of the largest prefix weighted by queue depth, every source is ranked by the wait it adds (`sourceWait` if busy) plus the recompute the computing pod pays for the tokens it holds short of the best-cached source, `(maxCached - cached) * (prefill - transfer)`. Sources within one block's recompute of the minimum are sampled uniformly by request-ID hash. The source is chosen before the computing pod is known, so the ranking prices the shortfall without the `fleetWeight` credit; with `sourceWaitMs: 0` this does not change the ranking, and with a non-zero `sourceWaitMs` it undervalues cache for a loaded computing pod.

Parameters:

- `prefillMicrosecondsPerToken` (float, required): prefill cost per token on the computing pod.
- `transferMicrosecondsPerToken` (float, required): P2P transfer cost per token; must be below the prefill cost.
- `transferFixedMs` (float, optional, default `0`): fixed cost of a pull.
- `sourceWaitMs` (float, optional, default `0`): added when the source is busy. Pulls are served from the source's CPU tier, not its scheduler; a busy source measured no added pull latency on the fleets below.
- `requeueMs` (float, optional, default `0`): added when the computing pod has a waiting queue.
- `fleetWeight` (float, optional, default `0`): credit per running request on the computing pod, in units of the prefill time the pull saves.
- `busyQueueThreshold` (int, optional, default `1`): waiting-queue depth at or above which a pod counts as busy.

`prefillMicrosecondsPerToken` and `transferMicrosecondsPerToken` come from an idle pull-versus-recompute ladder (the `calibrate-min-cached-token-delta.sh` recipe in the llm-d guides prints both). `transferFixedMs` and `requeueMs` are larger under load than on an idle pod pair, because the engine re-queues a pulled request after its blocks arrive; measure them on a loaded fleet. Values measured for gpt-oss-120b, TP=1:

| Fleet | prefill | transfer | transferFixed | requeue | sourceWait |
| --- | --- | --- | --- | --- | --- |
| H200, pull over TCP | 33 | 16 | 400 | 400 | 0 |
| B200, pull over RDMA | 19 | 10 | 200 | 125 | 0 |

With `fleetWeight: 1`, a pod with a few requests running pulls any delta above the floor, and an idle pod pulls only deltas whose own prefill saving exceeds the fixed cost (about 23.5K tokens with the H200 values).

```yaml
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      minCachedTokenDelta: 256
      costModel:
        prefillMicrosecondsPerToken: 33
        transferMicrosecondsPerToken: 16
        transferFixedMs: 400
        requeueMs: 400
        fleetWeight: 1
```

## Deployment Requirements

The emitted header only results in a KV transfer when the serving pods are
configured to serve and pull blocks over the P2P tier:

- vLLM runs the `OffloadingConnector` with a `p2p` secondary tier, and the routing sidecar consumes the header to inject the pull.
- `offload_prompt_only: false` in `kv_connector_extra_config` on any pod whose cache may be pulled — with the default (`true`), decode-phase (generated) blocks are never offloaded, so a pull of that content misses.
- Identical `--block-size` across peers; a mismatch makes vLLM reject the transfer (`block_len mismatch`).
- Identical `PYTHONHASHSEED` across peers, so block hashes match across processes.

---

## Related Documentation
- [Approximate Prefix Cache Producer](../approximateprefix/README.md)
- [Precise Prefix Cache Producer](../preciseprefixcache/README.md)
