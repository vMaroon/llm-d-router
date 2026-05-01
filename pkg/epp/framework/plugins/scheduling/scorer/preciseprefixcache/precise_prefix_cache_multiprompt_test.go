/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package preciseprefixcache

import (
	"context"
	"fmt"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

// fakeBlockKeys returns n synthetic kvblock.BlockHash entries — used to feed
// the mock indexer's totalBlocks accounting in absolute-normalization tests.
// The contents don't matter; only the length is read by getScores.
func fakeBlockKeys(n int) []kvblock.BlockHash {
	out := make([]kvblock.BlockHash, n)
	for i := range out {
		out[i] = kvblock.BlockHash(uint64(i + 1))
	}
	return out
}

// TestScorer_MultiPromptAggregatesHits verifies that when a Completions
// request carries an OpenAI-style array of prompts, the precise scorer issues
// one tokenization/scoring call per prompt, sums the per-pod hits AND the
// per-prompt block totals, and emits absolute coverage scores in [0, 1].
func TestScorer_MultiPromptAggregatesHits(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// Per-prompt scores: pod-a=5+1=6, pod-b=2+4=6 (tie).
	// Per-prompt totalBlocks: 10 each → totalBlocks = 20.
	// Expected absolute: pod-a = 6/20 = 0.3, pod-b = 6/20 = 0.3.
	perPromptScores := map[string]map[string]float64{
		"prompt-zero": {"10.0.0.1:8080": 5, "10.0.0.2:8080": 2},
		"prompt-one":  {"10.0.0.1:8080": 1, "10.0.0.2:8080": 4},
	}

	var seenPrompts []string
	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, prompt, _ string, _ []string) (map[string]float64, error) {
				seenPrompts = append(seenPrompts, prompt)
				if s, ok := perPromptScores[prompt]; ok {
					return s, nil
				}
				return nil, fmt.Errorf("unexpected prompt: %s", prompt)
			},
			computeBlockKeysFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string) ([]kvblock.BlockHash, error) {
				return fakeBlockKeys(10), nil
			},
		},
	}

	request := &scheduling.LLMRequest{
		RequestId:   "test-multi-prompt-equal",
		TargetModel: "test-model",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: scheduling.Prompt{Strings: []string{"prompt-zero", "prompt-one"}},
			},
		},
	}

	got := scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)
	require.NotEmpty(t, got)
	require.ElementsMatch(t, []string{"prompt-zero", "prompt-one"}, seenPrompts,
		"each prompt in the array must be tokenized exactly once")

	gotByAddr := make(map[string]float64)
	for ep, score := range got {
		m := ep.GetMetadata()
		gotByAddr[fmt.Sprintf("%s:%s", m.Address, m.Port)] = score
	}
	assert.InDelta(t, 0.3, gotByAddr["10.0.0.1:8080"], 1e-9)
	assert.InDelta(t, 0.3, gotByAddr["10.0.0.2:8080"], 1e-9)
}

// TestScorer_MultiPromptPicksHighestSum verifies absolute-normalized scores
// preserve the relative ordering of cumulative hit counts and that the
// loser is not stretched up to 1.0 (the old min-max behavior).
func TestScorer_MultiPromptPicksHighestSum(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// pod-a: 5+5=10 hits, pod-b: 1+2=3 hits.
	// totalBlocks per prompt = 10 → totalBlocks = 20.
	// Absolute: pod-a = 10/20 = 0.5, pod-b = 3/20 = 0.15.
	perPromptScores := map[string]map[string]float64{
		"p0": {"10.0.0.1:8080": 5, "10.0.0.2:8080": 1},
		"p1": {"10.0.0.1:8080": 5, "10.0.0.2:8080": 2},
	}

	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, prompt, _ string, _ []string) (map[string]float64, error) {
				return perPromptScores[prompt], nil
			},
			computeBlockKeysFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string) ([]kvblock.BlockHash, error) {
				return fakeBlockKeys(10), nil
			},
		},
	}

	request := &scheduling.LLMRequest{
		RequestId:   "test-multi-prompt-winner",
		TargetModel: "test-model",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: scheduling.Prompt{Strings: []string{"p0", "p1"}},
			},
		},
	}

	got := scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)
	require.NotEmpty(t, got)

	gotByAddr := make(map[string]float64)
	for ep, score := range got {
		m := ep.GetMetadata()
		gotByAddr[fmt.Sprintf("%s:%s", m.Address, m.Port)] = score
	}
	assert.InDelta(t, 0.5, gotByAddr["10.0.0.1:8080"], 1e-9, "pod-a 10 hits / 20 blocks")
	assert.InDelta(t, 0.15, gotByAddr["10.0.0.2:8080"], 1e-9, "pod-b 3 hits / 20 blocks (no longer stretched to 0.0)")
}

// TestScorer_SinglePromptAbsoluteNormalization confirms the single-prompt
// path also uses absolute coverage and only issues one GetPodScores call.
func TestScorer_SinglePromptAbsoluteNormalization(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scoreCalls := 0
	blockCalls := 0
	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, prompt, _ string, _ []string) (map[string]float64, error) {
				scoreCalls++
				assert.Equal(t, "hello", prompt)
				return map[string]float64{"10.0.0.1:8080": 4}, nil
			},
			computeBlockKeysFunc: func(_ context.Context, _ *types.RenderChatRequest, prompt, _ string) ([]kvblock.BlockHash, error) {
				blockCalls++
				assert.Equal(t, "hello", prompt)
				return fakeBlockKeys(8), nil
			},
		},
	}

	request := &scheduling.LLMRequest{
		RequestId:   "test-single",
		TargetModel: "test-model",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{Prompt: scheduling.Prompt{Raw: "hello"}},
		},
	}

	got := scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)
	assert.Equal(t, 1, scoreCalls, "single-prompt requests must issue exactly one GetPodScores call")
	assert.Equal(t, 1, blockCalls, "single-prompt requests must issue exactly one ComputeBlockKeys call")

	gotByAddr := make(map[string]float64)
	for ep, score := range got {
		m := ep.GetMetadata()
		gotByAddr[fmt.Sprintf("%s:%s", m.Address, m.Port)] = score
	}
	// pod-a: 4 hits / 8 blocks = 0.5. The other endpoint in testEndpoints
	// has no hits → 0.0 (cold pod no longer gets stretched to 0/0).
	assert.InDelta(t, 0.5, gotByAddr["10.0.0.1:8080"], 1e-9)
}

// TestScorer_ColdClusterReturnsZero is the regression guard: with no hits
// reported for any pod, absolute normalization must yield 0.0 across the
// board, not the 1.0 uniform that the old min-max code produced.
func TestScorer_ColdClusterReturnsZero(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string, _ []string) (map[string]float64, error) {
				return map[string]float64{}, nil
			},
			computeBlockKeysFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string) ([]kvblock.BlockHash, error) {
				return fakeBlockKeys(8), nil
			},
		},
	}

	request := &scheduling.LLMRequest{
		RequestId:   "test-cold",
		TargetModel: "test-model",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{Prompt: scheduling.Prompt{Raw: "hello"}},
		},
	}

	got := scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)
	require.NotEmpty(t, got)
	for _, score := range got {
		assert.Equal(t, 0.0, score, "cold cluster must score 0.0, not 1.0 (old min-max bug)")
	}
}

// TestScorer_MultiPromptCycleStateFastPath verifies that when the tokenizer
// plugin has pre-tokenized a multi-prompt request, the precise scorer reads
// TokenIDsList from CycleState, calls ScoreTokens once per sequence with
// nil MM features, and aggregates both scores AND totalBlocks across
// sequences. No tokenization should happen here — this is the fast path.
func TestScorer_MultiPromptCycleStateFastPath(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// Block size 4. Two sequences of 8 and 12 tokens → totalBlocks = 2 + 3 = 5.
	seq0 := []uint32{1, 2, 3, 4, 5, 6, 7, 8}
	seq1 := []uint32{11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}

	type call struct {
		tokens []uint32
		extra  []*kvblock.BlockExtraFeatures
	}
	var calls []call
	scoreTokens := 0
	scorer := &Scorer{
		typedName:       plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig:  &kvevents.Config{},
		pluginState:     plugin.NewPluginState(ctx),
		blockSizeTokens: 4,
		kvCacheIndexer: &mockKVCacheIndexer{
			scoreTokensFunc: func(_ context.Context, tokens []uint32, _ string, _ []string, extra []*kvblock.BlockExtraFeatures) (map[string]float64, error) {
				scoreTokens++
				calls = append(calls, call{tokens: tokens, extra: extra})
				// pod-a gets +2 hits per prompt, pod-b gets +1 per prompt.
				return map[string]float64{
					"10.0.0.1:8080": 2,
					"10.0.0.2:8080": 1,
				}, nil
			},
			computeBlockKeysFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string) ([]kvblock.BlockHash, error) {
				t.Fatal("ComputeBlockKeys must NOT be called when CycleState fast path is taken")
				return nil, nil
			},
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, _, _ string, _ []string) (map[string]float64, error) {
				t.Fatal("GetPodScores must NOT be called when CycleState fast path is taken")
				return nil, nil
			},
		},
	}

	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDsList: [][]uint32{seq0, seq1},
	})

	request := &scheduling.LLMRequest{
		RequestId:   "test-multi-cycle",
		TargetModel: "test-model",
	}

	got := scorer.Score(ctx, cycleState, request, testEndpoints)
	require.NotEmpty(t, got)

	// One ScoreTokens call per sequence, in order, with no MM features.
	require.Equal(t, 2, scoreTokens, "ScoreTokens must be called once per sequence")
	require.Len(t, calls, 2)
	assert.Equal(t, seq0, calls[0].tokens)
	assert.Nil(t, calls[0].extra, "multi-prompt completions are text-only")
	assert.Equal(t, seq1, calls[1].tokens)
	assert.Nil(t, calls[1].extra)

	// Aggregated: pod-a=4 hits, pod-b=2 hits, totalBlocks=5.
	// Absolute: pod-a = 4/5 = 0.8, pod-b = 2/5 = 0.4.
	gotByAddr := make(map[string]float64)
	for ep, score := range got {
		m := ep.GetMetadata()
		gotByAddr[fmt.Sprintf("%s:%s", m.Address, m.Port)] = score
	}
	assert.InDelta(t, 0.8, gotByAddr["10.0.0.1:8080"], 1e-9, "pod-a 4 hits / 5 totalBlocks")
	assert.InDelta(t, 0.4, gotByAddr["10.0.0.2:8080"], 1e-9, "pod-b 2 hits / 5 totalBlocks")
}
