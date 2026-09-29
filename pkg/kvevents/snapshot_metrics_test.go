/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvevents

import (
	"context"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

const (
	indexAdmissionsMetric = "llm_d_epp_kv_cache_index_admissions_total"
	indexEvictionsMetric  = "llm_d_epp_kv_cache_index_evictions_total"
	indexLookupsMetric    = "llm_d_epp_kv_cache_index_lookup_requests_total"
	indexHitsMetric       = "llm_d_epp_kv_cache_index_lookup_hits_total"
	indexLatencyMetric    = "llm_d_epp_kv_cache_index_lookup_latency_seconds"
)

// With enableMetrics, the index the snapshot manager writes and scores against
// reports the KV events its publisher generations apply and the scored lookups
// routing runs against it, on the registry the EPP serves at /metrics.
func TestSnapshotIndexMetricsCountEventsAndScoredLookups(t *testing.T) {
	ctx := context.Background()
	tokens, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 4, HashSeed: "test"})
	require.NoError(t, err)
	indexCfg := kvblock.DefaultIndexConfig()
	indexCfg.EnableMetrics = true
	cfg := DefaultConfig()
	cfg.SnapshotPort = 6000
	manager, err := NewSnapshotManager(cfg, snapshotTestIndex(t, indexCfg), tokens, snapshotRankAdapter{})
	require.NoError(t, err)

	admissions := registryCounter(t, indexAdmissionsMetric)
	evictions := registryCounter(t, indexEvictionsMetric)
	lookups := registryCounter(t, indexLookupsMetric)
	hits := registryCounter(t, indexHitsMetric)
	latencySamples := registryMetric(t, indexLatencyMetric).GetHistogram().GetSampleCount()

	// A publisher generation applies events to the shared index, as snapshot
	// bootstrap and the live follow do.
	const (
		source     = "10.0.0.1:8000"
		generation = source + "#snapshot-1"
	)
	pool := newGenerationPool(manager.index, tokens, nil)
	defer pool.queues[0].ShutDown()
	pool.strict = true
	prompt := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	require.NoError(t, pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockStoredEvent{BlockHashes: []uint64{101, 102}, Tokens: prompt, BlockSize: 4, DeviceTier: "GPU"},
	}}, generation, "model"))
	assert.Equal(t, admissions+2, registryCounter(t, indexAdmissionsMetric), "two stored blocks")

	subscriber := &snapshotSubscriber{sourceEndpoint: source, activeID: generation}
	subscriber.lastReceive.Store(time.Now().UnixNano())
	manager.subscribers[source] = subscriber
	keys, err := tokens.TokensToKVBlockKeys(0, prompt, "model", nil)
	require.NoError(t, err)
	matches, err := manager.ScoredLookup(ctx, keys, sets.New(source), nil)
	require.NoError(t, err)
	require.Equal(t, 2, matches[source].MatchedBlocks)
	assert.Equal(t, lookups+1, registryCounter(t, indexLookupsMetric), "one scored lookup")
	assert.Equal(t, hits+2, registryCounter(t, indexHitsMetric), "best pod matched two blocks")
	assert.Equal(t, latencySamples+1, registryMetric(t, indexLatencyMetric).GetHistogram().GetSampleCount())

	require.NoError(t, pool.processEventBatch(ctx, &EventBatch{Events: []GenericEvent{
		&BlockRemovedEvent{BlockHashes: []uint64{102}, DeviceTier: "GPU"},
	}}, generation, "model"))
	assert.Equal(t, evictions+1, registryCounter(t, indexEvictionsMetric), "one removed block")
}

func registryCounter(t *testing.T, name string) float64 {
	t.Helper()
	return registryMetric(t, name).GetCounter().GetValue()
}

// registryMetric reads an unlabeled metric from the controller-runtime registry
// the EPP serves at /metrics. Each index collector also emits a deprecated
// kvcache_* twin, so testutil.ToFloat64 cannot read it; the current series is
// selected by name here.
func registryMetric(t *testing.T, name string) *dto.Metric {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() == name {
			require.Len(t, family.GetMetric(), 1)
			return family.GetMetric()[0]
		}
	}
	require.Failf(t, "metric not registered", "%s is not on the EPP registry", name)
	return nil
}
