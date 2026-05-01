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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

func TestCompletionPrompts(t *testing.T) {
	tests := []struct {
		name string
		in   scheduling.Prompt
		want []string
	}{
		{name: "empty", in: scheduling.Prompt{}, want: nil},
		{name: "raw only", in: scheduling.Prompt{Raw: "hello"}, want: []string{"hello"}},
		{
			name: "raw takes precedence over strings",
			in:   scheduling.Prompt{Raw: "hello", Strings: []string{"a", "b"}},
			want: []string{"hello"},
		},
		{
			name: "strings array",
			in:   scheduling.Prompt{Strings: []string{"a", "b", "c"}},
			want: []string{"a", "b", "c"},
		},
		{
			name: "strings drop empties",
			in:   scheduling.Prompt{Strings: []string{"a", "", "c"}},
			want: []string{"a", "c"},
		},
		{
			name: "strings all empty",
			in:   scheduling.Prompt{Strings: []string{"", ""}},
			want: []string{},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := completionPrompts(tc.in)
			assert.Equal(t, tc.want, got)
		})
	}
}

// TestScorer_MultiPromptAggregatesHits verifies that when a Completions
// request carries an OpenAI-style array of prompts, the precise scorer issues
// one tokenization/scoring call per prompt and sums the per-pod hits before
// normalization.
func TestScorer_MultiPromptAggregatesHits(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// Per-prompt scores: pod-a wins prompt-0 by 5, pod-b wins prompt-1 by 4.
	// Aggregated: pod-a=5+1=6, pod-b=2+4=6. They tie, so under min-max
	// normalization (min==max branch) both endpoints score 1.0 — confirming
	// the sum lands the same value on both pods.
	perPrompt := map[string]map[string]float64{
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
				if s, ok := perPrompt[prompt]; ok {
					return s, nil
				}
				return nil, fmt.Errorf("unexpected prompt: %s", prompt)
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
	assert.Equal(t, 1.0, gotByAddr["10.0.0.1:8080"])
	assert.Equal(t, 1.0, gotByAddr["10.0.0.2:8080"])
}

// TestScorer_MultiPromptPicksHighestSum verifies that when one pod has more
// total hits across the prompt array than the other, the aggregated score
// pushes that pod to 1.0 and the loser to 0.0 after normalization.
func TestScorer_MultiPromptPicksHighestSum(t *testing.T) {
	ctx := utils.NewTestContext(t)

	// pod-a: 5+5=10, pod-b: 1+2=3. pod-a should win.
	perPrompt := map[string]map[string]float64{
		"p0": {"10.0.0.1:8080": 5, "10.0.0.2:8080": 1},
		"p1": {"10.0.0.1:8080": 5, "10.0.0.2:8080": 2},
	}

	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, prompt, _ string, _ []string) (map[string]float64, error) {
				return perPrompt[prompt], nil
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
	assert.Equal(t, 1.0, gotByAddr["10.0.0.1:8080"], "pod-a has the higher cumulative hit count")
	assert.Equal(t, 0.0, gotByAddr["10.0.0.2:8080"])
}

// TestScorer_SinglePromptStillUsesRawPath verifies the single-string path
// remains a single GetPodScores call (no aggregation overhead).
func TestScorer_SinglePromptStillUsesRawPath(t *testing.T) {
	ctx := utils.NewTestContext(t)
	calls := 0
	scorer := &Scorer{
		typedName:      plugin.TypedName{Type: PrecisePrefixCachePluginType, Name: "test"},
		kvEventsConfig: &kvevents.Config{},
		pluginState:    plugin.NewPluginState(ctx),
		kvCacheIndexer: &mockKVCacheIndexer{
			getPodScoresFunc: func(_ context.Context, _ *types.RenderChatRequest, prompt, _ string, _ []string) (map[string]float64, error) {
				calls++
				assert.Equal(t, "hello", prompt)
				return map[string]float64{"10.0.0.1:8080": 1.0}, nil
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

	scorer.Score(ctx, scheduling.NewCycleState(), request, testEndpoints)
	assert.Equal(t, 1, calls, "single-prompt requests must issue exactly one tokenization call")
}
