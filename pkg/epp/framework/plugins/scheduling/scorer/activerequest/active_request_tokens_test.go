/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package activerequest

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name string
		req  *scheduling.LLMRequest
		want int
	}{
		{name: "nil request", req: nil, want: 1},
		{name: "nil body", req: &scheduling.LLMRequest{}, want: 1},
		{name: "empty completions", req: newCompletionsRequest("e", ""), want: 1},
		// "hello" = 5 chars, ceil(5/4) = 2.
		{name: "single short prompt", req: newCompletionsRequest("s", "hello"), want: 2},
		// 16 chars, ceil(16/4) = 4.
		{name: "single 16-char prompt", req: newCompletionsRequest("s2", "abcdefghijklmnop"), want: 4},
		// "abcd"+"efgh" = 8 chars, ceil(8/4) = 2.
		{name: "multi prompts sum chars", req: newCompletionsRequest("m", "", "abcd", "efgh"), want: 2},
		// chat: "hi" = 2 chars, ceil(2/4) = 1.
		{name: "chat completions short", req: newChatRequest("c"), want: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, estimateTokens(tc.req))
		})
	}
}

// TestPreRequest_TokensMode_UsesCycleStateTokens verifies the happy path:
// the tokenizer plugin has populated CycleState; Score stashes the token
// total; PreRequest consumes it as the load delta.
func TestPreRequest_TokensMode_UsesCycleStateTokens(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{LoadUnit: LoadUnitTokens})

	endpointA := newTestEndpoint("pod-a", 0)
	req := newCompletionsRequest("tok-1", "hello")

	// Simulate the tokenizer plugin's CycleState write (1234 tokens).
	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDs: make([]uint32, 1234),
	})

	// Score must run before PreRequest so the token total is stashed.
	scorer.Score(ctx, cycleState, req, []scheduling.Endpoint{endpointA})

	scorer.PreRequest(ctx, req, newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA}))

	assert.Equal(t, 1234, scorer.getPodCount("default/pod-a"),
		"PreRequest must consume the exact token total from Score's CycleState read")
}

// TestPreRequest_TokensMode_MultiPromptUsesTotalTokens verifies multi-prompt
// requests sum tokens across prompts (TotalTokens via TokenIDsList).
func TestPreRequest_TokensMode_MultiPromptUsesTotalTokens(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{LoadUnit: LoadUnitTokens})

	endpointA := newTestEndpoint("pod-a", 0)
	req := newCompletionsRequest("multi-tok", "", "p1", "p2", "p3")

	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		// Three prompts, 100 + 200 + 300 = 600 tokens total.
		TokenIDsList: [][]uint32{
			make([]uint32, 100),
			make([]uint32, 200),
			make([]uint32, 300),
		},
	})

	scorer.Score(ctx, cycleState, req, []scheduling.Endpoint{endpointA})
	scorer.PreRequest(ctx, req, newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA}))

	assert.Equal(t, 600, scorer.getPodCount("default/pod-a"),
		"multi-prompt must sum tokens via TotalTokens, not per-prompt count")
}

// TestPreRequest_TokensMode_FallsBackToCharEstimate verifies the
// chars-per-4 fallback is used when CycleState is empty (e.g. tokenizer
// plugin disabled or render failed) — the request still contributes a
// meaningful, non-zero load.
func TestPreRequest_TokensMode_FallsBackToCharEstimate(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{LoadUnit: LoadUnitTokens})

	endpointA := newTestEndpoint("pod-a", 0)
	// 100 chars / 4 = 25 estimated tokens.
	prompt := make([]byte, 100)
	for i := range prompt {
		prompt[i] = 'x'
	}
	req := newCompletionsRequest("fallback-tok", string(prompt))

	// No Score call → CycleState stash empty → fallback path.
	scorer.PreRequest(ctx, req, newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA}))

	assert.Equal(t, 25, scorer.getPodCount("default/pod-a"),
		"with no CycleState tokens, fallback must use chars/4")
}

// TestPreRequest_TokensMode_ExactWinsOverFallback verifies that when both
// the CycleState stash AND a char estimate are available, the exact
// CycleState value wins. Critical for accuracy — char estimate exists only
// as a backstop.
func TestPreRequest_TokensMode_ExactWinsOverFallback(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{LoadUnit: LoadUnitTokens})

	endpointA := newTestEndpoint("pod-a", 0)
	// 400 chars; estimate would be 100. Real total: 47.
	prompt := make([]byte, 400)
	for i := range prompt {
		prompt[i] = 'y'
	}
	req := newCompletionsRequest("exact-vs-est", string(prompt))

	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDs: make([]uint32, 47),
	})

	scorer.Score(ctx, cycleState, req, []scheduling.Endpoint{endpointA})
	scorer.PreRequest(ctx, req, newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA}))

	assert.Equal(t, 47, scorer.getPodCount("default/pod-a"),
		"exact CycleState count must override the char estimate")
}

// TestResponseBody_TokensMode_DrainsByExactCount verifies that completing a
// tokens-mode request fully drains its load, using the Count stored on
// the request entry (which equals the delta increment used at PreRequest).
func TestResponseBody_TokensMode_DrainsByExactCount(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{LoadUnit: LoadUnitTokens})

	endpointA := newTestEndpoint("pod-a", 0)
	req := newCompletionsRequest("drain-tok", "hello")
	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDs: make([]uint32, 5000),
	})

	scorer.Score(ctx, cycleState, req, []scheduling.Endpoint{endpointA})
	scorer.PreRequest(ctx, req, newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA}))
	require.Equal(t, 5000, scorer.getPodCount("default/pod-a"))

	scorer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, endpointA.GetMetadata())

	assert.False(t, scorer.hasPodCount("default/pod-a"),
		"response complete must subtract exactly 5000, fully draining the pod")
}

// TestPreRequest_TokensMode_MixedExactAndFallback verifies that two
// concurrent requests on the same pod — one with exact tokens, one
// fallback to char estimate — produce a coherent total.
func TestPreRequest_TokensMode_MixedExactAndFallback(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, &Parameters{LoadUnit: LoadUnitTokens})
	endpointA := newTestEndpoint("pod-a", 0)
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA})

	// Request 1: exact tokens via CycleState.
	r1 := newCompletionsRequest("r1", "hi")
	cycleState1 := scheduling.NewCycleState()
	cycleState1.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDs: make([]uint32, 800),
	})
	scorer.Score(ctx, cycleState1, r1, []scheduling.Endpoint{endpointA})
	scorer.PreRequest(ctx, r1, result)

	// Request 2: no tokenizer → fallback. 200 chars / 4 = 50 estimate.
	prompt := make([]byte, 200)
	for i := range prompt {
		prompt[i] = 'z'
	}
	r2 := newCompletionsRequest("r2", string(prompt))
	scorer.PreRequest(ctx, r2, result)

	assert.Equal(t, 800+50, scorer.getPodCount("default/pod-a"),
		"exact + fallback must sum coherently in the same unit (tokens)")
}

// TestRequestsModeUnchanged is a backward-compat regression: with default
// LoadUnit (requests), CycleState tokens are ignored and counts stay in
// the original "requests" unit.
func TestRequestsModeUnchanged(t *testing.T) {
	ctx := utils.NewTestContext(t)
	// No LoadUnit set → defaults to "requests".
	scorer := NewActiveRequest(ctx, &Parameters{})

	endpointA := newTestEndpoint("pod-a", 0)
	req := newCompletionsRequest("backcompat", "hello")
	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDs: make([]uint32, 9999),
	})

	scorer.Score(ctx, cycleState, req, []scheduling.Endpoint{endpointA})
	scorer.PreRequest(ctx, req, newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA}))

	assert.Equal(t, 1, scorer.getPodCount("default/pod-a"),
		"requests mode must ignore CycleState tokens and increment by promptCount (1)")
}
