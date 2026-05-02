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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

func newCompletionsRequest(id, raw string, strs ...string) *scheduling.LLMRequest {
	prompt := scheduling.Prompt{}
	if raw != "" {
		prompt.Raw = raw
	}
	if len(strs) > 0 {
		prompt.Strings = strs
	}
	return &scheduling.LLMRequest{
		RequestId: id,
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{Prompt: prompt},
		},
	}
}

func newChatRequest(id string) *scheduling.LLMRequest {
	return &scheduling.LLMRequest{
		RequestId: id,
		Body: &scheduling.LLMRequestBody{
			ChatCompletions: &scheduling.ChatCompletionsRequest{
				Messages: []scheduling.Message{{Role: "user", Content: scheduling.Content{Raw: "hi"}}},
			},
		},
	}
}

func TestPromptCount(t *testing.T) {
	tests := []struct {
		name string
		req  *scheduling.LLMRequest
		want int
	}{
		{name: "nil request", req: nil, want: 1},
		{name: "nil body", req: &scheduling.LLMRequest{}, want: 1},
		{name: "chat completions", req: newChatRequest("c1"), want: 1},
		{name: "single prompt raw", req: newCompletionsRequest("r1", "hello"), want: 1},
		{name: "empty raw and empty strings", req: newCompletionsRequest("r2", ""), want: 1},
		{name: "raw wins over strings", req: newCompletionsRequest("r3", "hi", "a", "b"), want: 1},
		{name: "two prompts", req: newCompletionsRequest("r4", "", "a", "b"), want: 2},
		{name: "three prompts", req: newCompletionsRequest("r5", "", "a", "b", "c"), want: 3},
		{name: "empty entries dropped", req: newCompletionsRequest("r6", "", "a", "", "c"), want: 2},
		{name: "all empty entries", req: newCompletionsRequest("r7", "", "", ""), want: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, promptCount(tc.req))
		})
	}
}

// TestPreRequest_MultiPromptIncrementsByN verifies that a multi-prompt
// completion loads the chosen pod by len(Prompt.Strings) instead of 1.
func TestPreRequest_MultiPromptIncrementsByN(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, nil)

	endpointA := newTestEndpoint("pod-a", 0)
	req := newCompletionsRequest("multi-1", "", "p1", "p2", "p3")
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{
		"default": endpointA,
	})

	scorer.PreRequest(ctx, req, result)

	assert.Equal(t, 3, scorer.getPodCount("default/pod-a"), "3-prompt request must load the pod by 3")
}

// TestPreRequest_MultiPromptMultipleEndpoints verifies that PreRequest
// increments each profile's target endpoint by N (e.g. P/D disagg routes
// the same multi-prompt request to both decode and prefill).
func TestPreRequest_MultiPromptMultipleEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, nil)

	endpointA := newTestEndpoint("pod-a", 0)
	endpointB := newTestEndpoint("pod-b", 0)
	req := newCompletionsRequest("multi-pd", "", "p1", "p2")
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{
		"default": endpointA,
		"prefill": endpointB,
	})

	scorer.PreRequest(ctx, req, result)

	assert.Equal(t, 2, scorer.getPodCount("default/pod-a"))
	assert.Equal(t, 2, scorer.getPodCount("default/pod-b"))
}

// TestResponseBody_MultiPromptDecrementsByN verifies that completing a
// multi-prompt request fully drains its load — increment and decrement
// always use the same delta because Count is stored on requestEntry.
func TestResponseBody_MultiPromptDecrementsByN(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, nil)
	endpointA := newTestEndpoint("pod-a", 0)

	req := newCompletionsRequest("multi-resp", "", "p1", "p2", "p3", "p4")
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA})

	scorer.PreRequest(ctx, req, result)
	require.Equal(t, 4, scorer.getPodCount("default/pod-a"))

	scorer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, endpointA.GetMetadata())

	assert.False(t, scorer.hasPodCount("default/pod-a"),
		"after response complete, pod count for a single-tenant pod must drop to 0")
}

// TestTTLExpiration_MultiPromptDecrementsByN verifies that the OnEviction
// callback uses the stored Count when a multi-prompt request is dropped on
// TTL — preventing under-decrement leaks for abandoned multi-prompt loads.
func TestTTLExpiration_MultiPromptDecrementsByN(t *testing.T) {
	ctx := utils.NewTestContext(t)
	params := &Parameters{RequestTimeout: "1s"}
	scorer := NewActiveRequest(ctx, params)
	endpointA := newTestEndpoint("pod-a", 0)

	req := newCompletionsRequest("multi-ttl", "", "p1", "p2", "p3")
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA})

	scorer.PreRequest(ctx, req, result)
	require.Equal(t, 3, scorer.getPodCount("default/pod-a"))

	time.Sleep(2 * time.Second)
	scorer.requestCache.DeleteExpired()

	assert.False(t, scorer.hasPodCount("default/pod-a"),
		"TTL eviction must decrement by the same N used at increment, fully draining the pod count")
}

// TestPreRequest_MixedSinglePlusMultiCount verifies that interleaved single
// and multi-prompt requests on the same pod sum correctly. Important because
// the scorer's running count is the routing signal — a single pod handling
// 1 multi-prompt(3) + 2 singletons should report 5, not 3.
func TestPreRequest_MixedSinglePlusMultiCount(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, nil)
	endpointA := newTestEndpoint("pod-a", 0)
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA})

	scorer.PreRequest(ctx, newCompletionsRequest("multi", "", "p1", "p2", "p3"), result) // +3
	scorer.PreRequest(ctx, newCompletionsRequest("single1", "x"), result)                 // +1
	scorer.PreRequest(ctx, newCompletionsRequest("single2", "y"), result)                 // +1

	assert.Equal(t, 5, scorer.getPodCount("default/pod-a"))
}

// TestResponseBody_PartialDrainKeepsRemainder verifies that completing one
// of three concurrent requests on the same pod leaves the other two intact.
// Specifically: if only the multi-prompt request completes, single-prompt
// loads from other requests must remain.
func TestResponseBody_PartialDrainKeepsRemainder(t *testing.T) {
	ctx := utils.NewTestContext(t)
	scorer := NewActiveRequest(ctx, nil)
	endpointA := newTestEndpoint("pod-a", 0)
	result := newTestSchedulingResult(map[string]scheduling.Endpoint{"default": endpointA})

	multi := newCompletionsRequest("multi", "", "p1", "p2", "p3")
	single1 := newCompletionsRequest("single1", "x")
	single2 := newCompletionsRequest("single2", "y")

	scorer.PreRequest(ctx, multi, result)
	scorer.PreRequest(ctx, single1, result)
	scorer.PreRequest(ctx, single2, result)
	require.Equal(t, 5, scorer.getPodCount("default/pod-a"))

	// Drain just the multi-prompt request.
	scorer.ResponseBody(ctx, multi, &requestcontrol.Response{EndOfStream: true}, endpointA.GetMetadata())

	assert.Equal(t, 2, scorer.getPodCount("default/pod-a"),
		"draining a 3-prompt request must subtract exactly 3, leaving the two singletons (=2)")
}
