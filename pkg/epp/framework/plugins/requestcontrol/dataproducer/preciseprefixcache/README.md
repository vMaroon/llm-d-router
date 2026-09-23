# Precise Prefix Cache Producer

**Type:** `precise-prefix-cache-producer`

DataProducer that owns the precise KV-block index and publishes
per-endpoint `PrefixCacheMatchInfo`. Pairs with the generic
[`prefix-cache-scorer`](../../../scheduling/scorer/prefix/); the scorer
must reference this producer by name:

```yaml
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
```

Without the `prefixMatchInfoProducerName` field, the scorer falls back
to the auto-spawned approx producer.

Pipeline per request:
- Consume `TokenizedPrompt` from `token-producer`.
- Hash tokens → KV-block keys → `kvblock.Index.Lookup`.
- Write `PrefixCacheMatchInfo(matchBlocks, totalBlocks, blockSizeTokens)` per endpoint, including the unweighted cached-block count and its per-device-tier breakdown.
- (`PreRequest`) Speculative-index the selected endpoint(s) with TTL eviction.
- (`EndpointExtractor`) Per-pod ZMQ subscriber lifecycle on add/delete.

Requires `TokenizedPrompt` on the request — set by a `token-producer`
upstream. No-op otherwise.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenProcessorConfig` | object | `kvblock.DefaultTokenProcessorConfig()` | KV-block hashing for the EPP-recomputed keys (block size, hash seed). |
| `indexerConfig` | object | `kvcache.NewDefaultConfig()` | `kvcache.Indexer` config. |
| `kvEventsConfig` | object | `kvevents.DefaultConfig()` | KV-events pool config. |
| `speculativeIndexing` | bool | `false` | Seed predicted entries on routing decisions. |
| `speculativeTTL` | duration | `2s` | TTL for speculative entries. |
| `fullReportRepair` | object | disabled | Enables bounded requests for authoritative per-request KV-cache reports. |
| `fullReportRepair.fullReportThreshold` | number | `0.80` | Request a full report when the selected endpoint's confirmed contiguous match is below this fraction. |
| `fullReportRepair.minMissingBlocks` | integer | `32` | Minimum confirmed-block deficit required before requesting a full report. |
| `fullReportRepair.prefillProfileName` | string | `prefill` | Disaggregation prefill profile whose selected endpoint is evaluated for repair. |

Full-report repair is opt-in. A newly attached endpoint is eligible for repair.
It requires per-pod discovery and rejects global `kvEventsConfig.zmqEndpoint`
mode because global stream identities cannot be joined to scheduler `address:port`
endpoint keys.
The producer adds `vllm_xargs.kv_cache_report_mode: full` only when at least
`minMissingBlocks` are absent and the confirmed, non-speculative match is below
`fullReportThreshold`. Missing-parent, sequence-discontinuity, and event-processing
faults bypass the ratio once for the next selected request. Eligibility clears only
after a known-empty cache reset or an authoritative replay snapshot; sending a full
report is an attempt, not proof that the endpoint is repaired.
The request argument is body-wide, so a disaggregated decoder may also compute a
full report; keep reports bounded and measure decoder overhead before enabling this
repair in production. Scope `kvEventsConfig.podDiscoveryConfig.podLabelSelector`
to prefiller endpoints when the precise index is intended to model prefill residency.

Set `kvEventsConfig.engineType` to `sglang` for SGLang KV-events. It defaults
to `vllm` when omitted.

See [llm-d-kv-cache/docs/configuration.md](https://github.com/llm-d/llm-d-kv-cache/blob/main/docs/configuration.md)
for nested parameter details.

## Engine compatibility

Block keys are recomputed by the EPP from `TokenizedPrompt` (tokens, model,
multimodal features, cache salt) on both the lookup path and the KV-event
ingestion path, using this plugin's `tokenProcessorConfig`. The engine's own
block hashes serve only as opaque keys for the engine-to-request mapping, so
`blockSize`/`hashSeed` need not match the engine.

The cross-engine requirement is that the engine emits, in its KV-events, the
hash-affecting inputs the EPP hashes: `token_ids`, and `extra_keys` carrying
multimodal identifiers and `cache_salt`. An input the engine omits from
`extra_keys` is absent on the event side, so requests carrying it do not
correlate.

| Engine | `extra_keys` in KV-events | `cache_salt` |
|--------|---------------------------|--------------|
| vLLM | emitted | in block-0 `extra_keys`; salted prefixes isolated and precise-routed |
| SGLang | not emitted | baked into engine block hashes but not surfaced; salted requests are precise-cache misses until SGLang emits `extra_keys` |

Salt isolation is enforced by the engine regardless; the above affects only
routing accuracy for salted requests.

## vLLM snapshot recovery

Set `kvEventsConfig.snapshotPort` to the vLLM snapshot service port. Zero disables
snapshot recovery. This requires the vLLM snapshot protocol with publisher UUIDs
and idle heartbeats, per-pod discovery, and the in-memory index. Each DP rank
uses `socketPort + RankIndex` for live events and `snapshotPort + RankIndex` for
snapshots. The two port ranges must not overlap and must remain at or below
65535. Replay, shared subscriber sockets, speculative indexing, and non-vLLM
engines are rejected. Events with locality or ownership scopes are rejected
until the router can preserve those dimensions.

The recovery index raises its per-key entry capacity to 1,048,576 so the
ordinary request index's small `podCacheSize` cannot silently discard snapshot
generations. Entries are allocated only when a publisher reports a matching
block; the configured key-count limit still bounds the number of indexed keys.
Each recovery generation keeps at most 1,048,576 engine-to-canonical mappings
so publishers cannot overwrite each other's reconstruction metadata. If a live
event references mapping history older than this bound, that generation is
removed from routing and rebuilt from a fresh snapshot.
At most four publishers fetch and install snapshots concurrently; additional
publishers remain non-routable until a recovery slot is available.

```yaml
- type: precise-prefix-cache-producer
  parameters:
    speculativeIndexing: false
    kvEventsConfig:
      discoverPods: true
      topicFilter: "kv@"
      snapshotPort: 6000
      podDiscoveryConfig:
        socketPort: 5557
- type: prefix-cache-scorer
  parameters:
    prefixMatchInfoProducerName: precise-prefix-cache-producer
```

Configure each engine's topic as `kv@<pod-IP>:<serving-port>@<model-name>`. The
model name must match the model requested through the EPP. Endpoint attribution
uses the discovered serving address. The live and snapshot ports must be
reachable from the EPP.

The producer subscribes before requesting a snapshot and builds a hidden
publisher generation in the shared index. It applies consecutive buffered live
events after the snapshot cut and activates the generation only after
reconstruction succeeds. A sequence gap, publisher UUID
change, malformed event, missing reconstruction metadata, or heartbeat timeout
removes that publisher's cache affinity and triggers another snapshot request.
Other publishers remain independently available. Endpoint deletion removes its
snapshot state. Ordinary routing remains available during recovery.

Snapshot recovery applies per publisher, only where the engine serves a
snapshot endpoint. The first live frame decides: vLLM appends its 16-byte
publisher identity to the sequence number, and heartbeats, only when
`snapshot_endpoint` is set. A publisher with 8-byte sequence frames is indexed
from live events alone, as without `snapshotPort`: its cache affinity is usable
immediately, idle periods are allowed, and sequence gaps are not recovered. A
change of frame format restarts the subscriber in the other mode, so engines and
the router can be upgraded or rolled back in any order.
`kv_cache_events_live_only_publishers` counts these publishers.

GPU resets preserve CPU residency. Duplicate stores retain their reference
counts, and tokenless offload events preserve every router block covered by an
engine block. The prefix scorer retains its configured device-tier weights.

Snapshot requests time out after 10 seconds. A publisher is unavailable after
5 seconds without a live message. Recovery retries start after 1 second and use
jittered exponential backoff capped at 30 seconds. Bootstrap buffering is
bounded to 4,096 messages and 64 MiB; the receive queue is bounded to 256
messages and 64 MiB. Snapshot replies are limited to 256 MiB. An unavailable
snapshot produces no cache affinity; if the engine recorder is invalid, the
publisher must be restarted to restore snapshots.

The cross-language regression starts the real vLLM publisher and recorder from
an installed feature checkout. It generates KV events without loading a model:

```bash
VLLM_PYTHON=/path/to/vllm/.venv/bin/python \
PYTHONPATH=/path/to/vllm \
VLLM_TEST_BLOCK_SIZE=8 \
go test -race ./pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache \
  -run TestSnapshotVLLMPublisher -count=1
```

The ordinary Go suite exercises recovery faults with controlled ZMQ publishers.
Neither test starts Envoy, Kubernetes, or GPU inference.
