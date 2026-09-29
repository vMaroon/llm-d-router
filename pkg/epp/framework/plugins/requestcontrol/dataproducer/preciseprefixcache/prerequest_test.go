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

package preciseprefixcache

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixmetrics"
	"github.com/llm-d/llm-d-router/test/utils"
)

// addCall captures one fakeKVBlockIndex.Add invocation.
type addCall struct {
	keys    []kvblock.BlockHash
	entries []kvblock.PodEntry
}

func newProducerForPreRequest(ctx context.Context, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	return newNamedProducerForPreRequest(ctx, "test", speculativeEnabled, idx)
}

func newNamedProducerForPreRequest(ctx context.Context, name string, speculativeEnabled bool, idx *fakeKVBlockIndex) *Producer {
	cache := ttlcache.New[string, *speculativeEntries](
		ttlcache.WithTTL[string, *speculativeEntries](time.Minute),
	)
	return &Producer{
		typedName:          plugin.TypedName{Type: PluginType, Name: name},
		kvCacheIndexer:     &fakeKVCacheIndexer{index: idx},
		dk:                 attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
		speculativeCache:   cache,
		speculativeTTL:     time.Minute,
		speculativeEnabled: speculativeEnabled,
		pluginState:        plugin.NewPluginState(ctx),
	}
}

func primaryOnly(name string, endpoint scheduling.Endpoint) *scheduling.SchedulingResult {
	return &scheduling.SchedulingResult{
		PrimaryProfileName: name,
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			name: {TargetEndpoints: []scheduling.Endpoint{endpoint}},
		},
	}
}

// speculativeEnabled=true with populated block keys: index.Add called once
// with the primary pod identifier, and a speculative cache entry is created.
func TestPreRequest_SeedsSpeculativeForPrimary(t *testing.T) {
	ctx := utils.NewTestContext(t)

	var calls []addCall
	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			calls = append(calls, addCall{keys: keys, entries: entries})
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	blockKeys := []kvblock.BlockHash{0xAA, 0xBB}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-1"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{blockKeys}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	require.Len(t, calls, 1)
	assert.Equal(t, blockKeys, calls[0].keys)
	require.Len(t, calls[0].entries, 1)
	assert.Equal(t, "10.0.0.1:8080", calls[0].entries[0].PodIdentifier)
	assert.True(t, calls[0].entries[0].Speculative)

	cached := p.speculativeCache.Get(req.RequestID)
	require.NotNil(t, cached)
	assert.Equal(t, [][]kvblock.BlockHash{blockKeys}, cached.Value().perPromptKeys)
	require.Len(t, cached.Value().podEntries, 1)
	assert.Equal(t, "10.0.0.1:8080", cached.Value().podEntries[0].PodIdentifier)
}

// speculativeEnabled=true with empty blockKeys: PreRequest must not call
// index.Add and must not create a cache entry.
func TestPreRequest_EmptyBlockKeys_NoAdd(t *testing.T) {
	ctx := utils.NewTestContext(t)

	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, _ []kvblock.BlockHash, _ []kvblock.PodEntry) error {
			t.Fatalf("index.Add must not be called when blockKeys are empty")
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	req := &scheduling.InferenceRequest{RequestID: "req-pre-empty"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: nil})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Nil(t, p.speculativeCache.Get(req.RequestID))
}

// P/D prefill profile: index.Add called twice (primary + prefill), and the
// cache entry tracks both pod identifiers.
func TestPreRequest_PrefillProfile_SeedsBoth(t *testing.T) {
	ctx := utils.NewTestContext(t)

	var calls []addCall
	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, keys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
			calls = append(calls, addCall{keys: keys, entries: entries})
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, true, idx)

	blockKeys := []kvblock.BlockHash{0xCC}
	req := &scheduling.InferenceRequest{RequestID: "req-pre-pd"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{perPromptKeys: [][]kvblock.BlockHash{blockKeys}})

	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":                   {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
	}
	_ = p.PreRequest(ctx, req, result)

	require.Len(t, calls, 2)
	assert.Equal(t, "10.0.0.1:8080", calls[0].entries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", calls[1].entries[0].PodIdentifier)

	cached := p.speculativeCache.Get(req.RequestID)
	require.NotNil(t, cached)
	require.Len(t, cached.Value().podEntries, 2)
	assert.Equal(t, "10.0.0.1:8080", cached.Value().podEntries[0].PodIdentifier)
	assert.Equal(t, "10.0.0.2:8080", cached.Value().podEntries[1].PodIdentifier)
}

// speculativeEnabled=false: early return — no index writes, no cache entry,
// and PluginState is left untouched.
func TestPreRequest_SpeculativeDisabled_NoOp(t *testing.T) {
	ctx := utils.NewTestContext(t)

	idx := &fakeKVBlockIndex{
		addFn: func(_ context.Context, _ []kvblock.BlockHash, _ []kvblock.BlockHash, _ []kvblock.PodEntry) error {
			t.Fatalf("index.Add must not be called when speculative indexing is disabled")
			return nil
		},
	}
	p := newProducerForPreRequest(ctx, false, idx)

	req := &scheduling.InferenceRequest{RequestID: "req-pre-off"}
	p.pluginState.Write(req.RequestID, blockKeysStateKey,
		&blockKeysState{perPromptKeys: [][]kvblock.BlockHash{{0xDD}}})

	_ = p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0]))

	assert.Nil(t, p.speculativeCache.Get(req.RequestID))
}

func TestPreRequest_FullReportRepairUsesConfiguredPrefillAndMergesXArgs(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, false, &fakeKVBlockIndex{})
	p.fullReportRepair = newFullReportRepair(FullReportRepairConfig{
		FullReportThreshold: 0.80,
		MinMissingBlocks:    32,
		PrefillProfileName:  "custom-prefill",
	})
	p.fullReportRepair.observe("10.0.0.2:8080", kvevents.StreamEventAttached)

	payload := fwkrh.PayloadMap{
		"model":      "model",
		"vllm_xargs": map[string]any{"existing": "preserved"},
	}
	req := &scheduling.InferenceRequest{
		RequestID: "repair-prefill",
		Body:      &fwkrh.InferenceRequestBody{Payload: payload},
	}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		repairMatches: map[string]repairMatch{
			"10.0.0.1:8080": {total: 200, confirmed: 200},
			"10.0.0.2:8080": {total: 200, confirmed: 159},
		},
	})
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":         {TargetEndpoints: []scheduling.Endpoint{testEndpoints[0]}},
			"custom-prefill": {TargetEndpoints: []scheduling.Endpoint{testEndpoints[1]}},
		},
	}

	require.NoError(t, p.PreRequest(ctx, req, result))
	xargs, ok := payload["vllm_xargs"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "preserved", xargs["existing"])
	assert.Equal(t, "full", xargs["kv_cache_report_mode"])
	again, _ := p.fullReportRepair.shouldRequest("10.0.0.2:8080", repairMatch{total: 200, confirmed: 159})
	assert.False(t, again, "a pending report must not be duplicated")
}

func TestPreRequest_FullReportRepairRepackagesJSONProtocols(t *testing.T) {
	tests := []struct {
		name    string
		payload fwkrh.PayloadMap
	}{
		{name: "completions", payload: fwkrh.PayloadMap{"prompt": "hello"}},
		{name: "chat completions", payload: fwkrh.PayloadMap{"messages": []any{map[string]any{"role": "user", "content": "hello"}}}},
		{name: "responses", payload: fwkrh.PayloadMap{"input": "hello"}},
		{name: "anthropic messages", payload: fwkrh.PayloadMap{"anthropic_version": "2023-06-01", "messages": []any{}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			p := newProducerForPreRequest(ctx, false, &fakeKVBlockIndex{})
			p.fullReportRepair = newFullReportRepair(FullReportRepairConfig{
				FullReportThreshold: 0.80,
				MinMissingBlocks:    32,
			})
			const endpoint = "10.0.0.1:8080"
			p.fullReportRepair.observe(endpoint, kvevents.StreamEventAttached)
			req := &scheduling.InferenceRequest{
				RequestID: tt.name,
				Body:      &fwkrh.InferenceRequestBody{Payload: tt.payload},
			}
			p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
				repairMatches: map[string]repairMatch{endpoint: {total: 200, confirmed: 100}},
			})

			require.NoError(t, p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0])))
			body, err := tt.payload.Marshal()
			require.NoError(t, err)
			var repackaged map[string]any
			require.NoError(t, json.Unmarshal(body, &repackaged))
			xargs, ok := repackaged["vllm_xargs"].(map[string]any)
			require.True(t, ok)
			assert.Equal(t, "full", xargs["kv_cache_report_mode"])
		})
	}
}

func TestPreRequest_FullReportRepairThresholdAndOneShotForce(t *testing.T) {
	tests := []struct {
		name      string
		total     int
		confirmed int
		signal    kvevents.StreamEvent
		wantFull  bool
	}{
		{name: "below threshold but too few missing", total: 100, confirmed: 79, wantFull: false},
		{name: "at threshold", total: 200, confirmed: 160, wantFull: false},
		{name: "materially under indexed", total: 200, confirmed: 159, wantFull: true},
		{name: "integrity signal bypasses ratio", total: 200, confirmed: 168,
			signal: kvevents.StreamEventMissingParent, wantFull: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := utils.NewTestContext(t)
			p := newProducerForPreRequest(ctx, false, &fakeKVBlockIndex{})
			p.fullReportRepair = newFullReportRepair(FullReportRepairConfig{
				FullReportThreshold: 0.80,
				MinMissingBlocks:    32,
			})
			const endpoint = "10.0.0.1:8080"
			p.fullReportRepair.observe(endpoint, kvevents.StreamEventAttached)
			if tt.signal != "" {
				p.fullReportRepair.observe(endpoint, tt.signal)
			}
			payload := fwkrh.PayloadMap{"model": "model"}
			req := &scheduling.InferenceRequest{
				RequestID: tt.name,
				Body:      &fwkrh.InferenceRequestBody{Payload: payload},
			}
			p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
				repairMatches: map[string]repairMatch{endpoint: {total: tt.total, confirmed: tt.confirmed}},
			})

			require.NoError(t, p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0])))
			xargs, _ := payload["vllm_xargs"].(map[string]any)
			assert.Equal(t, tt.wantFull, xargs["kv_cache_report_mode"] == "full")

			if tt.signal != "" {
				payload2 := fwkrh.PayloadMap{"model": "model"}
				req2 := &scheduling.InferenceRequest{
					RequestID: tt.name + "-second",
					Body:      &fwkrh.InferenceRequestBody{Payload: payload2},
				}
				p.pluginState.Write(req2.RequestID, blockKeysStateKey, &blockKeysState{
					repairMatches: map[string]repairMatch{endpoint: {total: tt.total, confirmed: tt.confirmed}},
				})
				require.NoError(t, p.PreRequest(ctx, req2, primaryOnly("default", testEndpoints[0])))
				xargs2, _ := payload2["vllm_xargs"].(map[string]any)
				assert.NotEqual(t, "full", xargs2["kv_cache_report_mode"],
					"the explicit bypass is consumed once")
			}
		})
	}
}

func TestPreRequest_FullReportRepairDoesNotOverwriteMalformedXArgs(t *testing.T) {
	ctx := utils.NewTestContext(t)
	p := newProducerForPreRequest(ctx, false, &fakeKVBlockIndex{})
	p.fullReportRepair = newFullReportRepair(FullReportRepairConfig{
		FullReportThreshold: 0.80,
		MinMissingBlocks:    32,
	})
	const endpoint = "10.0.0.1:8080"
	p.fullReportRepair.observe(endpoint, kvevents.StreamEventAttached)
	p.fullReportRepair.observe(endpoint, kvevents.StreamEventMissingParent)
	payload := fwkrh.PayloadMap{"vllm_xargs": "invalid"}
	req := &scheduling.InferenceRequest{
		RequestID: "malformed-xargs",
		Body:      &fwkrh.InferenceRequestBody{Payload: payload},
	}
	p.pluginState.Write(req.RequestID, blockKeysStateKey, &blockKeysState{
		repairMatches: map[string]repairMatch{endpoint: {total: 200, confirmed: 168}},
	})

	require.NoError(t, p.PreRequest(ctx, req, primaryOnly("default", testEndpoints[0])))
	assert.Equal(t, "invalid", payload["vllm_xargs"])
	request, reason := p.fullReportRepair.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 168})
	assert.True(t, request, "malformed request arguments must not consume the integrity bypass")
	assert.Equal(t, "integrity", reason)
}

func TestFullReportRepairLifecycle(t *testing.T) {
	r := newFullReportRepair(FullReportRepairConfig{FullReportThreshold: 0.80, MinMissingBlocks: 32})
	const endpoint = "10.0.0.1:8080"

	r.observe(endpoint, kvevents.StreamEventAttached)
	request, reason := r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 100})
	assert.True(t, request)
	assert.Equal(t, "threshold", reason)
	r.observe(endpoint, kvevents.StreamEventMissingParent)
	request, reason = r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 100})
	assert.False(t, request, "integrity event retains the pending report")
	assert.Empty(t, reason)
	request, reason = r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 100})
	assert.False(t, request)
	assert.Empty(t, reason)

	r.observe(endpoint, kvevents.StreamEventKnownEmpty)
	request, _ = r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 100})
	assert.False(t, request)
	r.observe(endpoint, kvevents.StreamEventAttached)
	r.observe(endpoint, kvevents.StreamEventAuthoritativeSnapshot)
	request, _ = r.shouldRequest(endpoint, repairMatch{total: 200, confirmed: 100})
	assert.False(t, request)
}

// The predicted token count comes from the chosen endpoint's unweighted cached
// block count, not the tier-weighted match score, and is reported with
// speculative indexing off alongside the prompt tokens it is measured against.
func TestPreRequest_RecordsPrediction(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-predicted-records"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	endpoint := freshEndpoints()[0]
	// Weighted score 2.5 against 4 cached blocks: the token count must follow
	// the cached count.
	endpoint.Put(p.dk, attrprefix.NewPrefixCacheMatchInfo(2, 8, testBlockSize).
		WithCachedBlockCount(4))

	// A non-default profile name proves the lookup follows PrimaryProfileName.
	beforePredicted := sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, promptTokensMetric, name).GetSampleSum()
	_ = p.PreRequest(ctx, tokenizedRequest("req-predicted", 8*testBlockSize),
		primaryOnly("decode", endpoint))

	assert.Equal(t, beforePredicted+float64(4*testBlockSize), sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleSum())
	assert.Equal(t, beforePrompt+float64(8*testBlockSize), sharedPrefixHistogram(t, promptTokensMetric, name).GetSampleSum())
}

// An endpoint the producer never published match info for is not observed:
// a zero would be indistinguishable from a real zero-hit prediction.
func TestPreRequest_NoMatchInfo_RecordsNothing(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const name = "precise-predicted-absent"
	p := newNamedProducerForPreRequest(ctx, name, false, &fakeKVBlockIndex{})

	before := sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleCount()
	_ = p.PreRequest(ctx, tokenizedRequest("req-no-info", testBlockSize),
		primaryOnly("default", freshEndpoints()[0]))
	assert.Equal(t, before, sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleCount())

	// No endpoint at all is equally a no-op.
	_ = p.PreRequest(ctx, tokenizedRequest("req-no-endpoint", testBlockSize),
		&scheduling.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults:     map[string]*scheduling.ProfileRunResult{"default": {}},
		})
	assert.Equal(t, before, sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleCount())
}

// Under P/D disaggregation the primary profile is decode, and only prefill
// endpoints publish KV events. Produce publishes match info on every candidate,
// so the decode target carries a zero-block match; the prediction must follow
// the prefill target that holds the cached blocks. The producer runs as in
// production: fused scored lookups, speculative indexing and full-report
// repair off.
func TestProduceThenPreRequest_PDRecordsPrefillPrediction(t *testing.T) {
	ctx := utils.NewTestContext(t)
	prefixmetrics.Register()

	const (
		name          = "precise-predicted-pd"
		decodeAddr    = "10.0.0.1:8080"
		prefillAddr   = "10.0.0.2:8080"
		promptBlocks  = 8
		prefillCached = 6
	)
	keys := make([]kvblock.BlockHash, promptBlocks)
	for i := range keys {
		keys[i] = kvblock.BlockHash(i + 1)
	}
	idx := &fakeKVCacheIndexer{
		computeFromTokens: func(context.Context, []uint32, string, []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error) {
			return keys, nil
		},
		index: &fakeScoredKVBlockIndex{
			fakeKVBlockIndex: &fakeKVBlockIndex{},
			scoredLookup: func(context.Context, []kvblock.BlockHash, sets.Set[string], map[string]float64,
			) (map[string]kvblock.PodMatchStats, error) {
				return map[string]kvblock.PodMatchStats{
					prefillAddr: {WeightedScore: prefillCached, MatchedBlocks: prefillCached, ConfirmedBlocks: prefillCached},
				}, nil
			},
		},
	}
	p := newProducerWithIndexer(ctx, idx, &fakeKVBlockScorer{})
	p.typedName.Name = name
	p.dk = attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name)
	require.False(t, p.speculativeEnabled)
	require.Nil(t, p.fullReportRepair)

	endpoints := freshEndpoints() // [0] = decode (10.0.0.1), [1] = prefill (10.0.0.2)
	req := tokenizedRequest("req-pd-predicted", promptBlocks*testBlockSize)
	req.TargetModel = "model"
	require.NoError(t, p.Produce(ctx, req, endpoints))

	decodeInfo, ok := endpoints[0].Get(p.dk)
	require.True(t, ok, "Produce publishes match info on the decode candidate too")
	require.Equal(t, 0, decodeInfo.(*attrprefix.PrefixCacheMatchInfo).CachedBlockCount())

	beforePredicted := sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleSum()
	beforePrompt := sharedPrefixHistogram(t, promptTokensMetric, name).GetSampleSum()
	beforeCount := sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleCount()
	require.NoError(t, p.PreRequest(ctx, req, &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":                   {TargetEndpoints: []scheduling.Endpoint{endpoints[0]}},
			experimentalPrefillProfile: {TargetEndpoints: []scheduling.Endpoint{endpoints[1]}},
		},
	}))

	assert.Equal(t, beforeCount+1, sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleCount())
	assert.Equal(t, beforePredicted+float64(prefillCached*testBlockSize),
		sharedPrefixHistogram(t, predictedCachedTokensMetric, name).GetSampleSum(),
		"prediction must be the prefill target's cached tokens, not the decode target's 0")
	assert.Equal(t, beforePrompt+float64(promptBlocks*testBlockSize),
		sharedPrefixHistogram(t, promptTokensMetric, name).GetSampleSum())
}

func tokenizedRequest(id string, tokenCount int) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{
		RequestID: id,
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{make([]uint32, tokenCount)}},
		},
	}
}

const (
	predictedCachedTokensMetric = "llm_d_epp_prefix_predicted_cached_tokens"
	promptTokensMetric          = "llm_d_epp_prefix_prompt_tokens"
)

// sharedPrefixHistogram reads a shared prefix metric out of the registry it is
// registered against, since those metrics live in another package. A metric
// that has not been observed yet reads as nil, whose accessors return zero.
func sharedPrefixHistogram(t *testing.T, metricName, pluginName string) *dto.Histogram {
	t.Helper()
	families, err := ctrlmetrics.Registry.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != metricName {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "plugin_name" && label.GetValue() == pluginName {
					return metric.GetHistogram()
				}
			}
		}
	}
	return nil
}
