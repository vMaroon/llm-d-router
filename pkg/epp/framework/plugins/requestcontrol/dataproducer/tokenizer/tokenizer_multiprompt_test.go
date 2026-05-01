//go:build !gaie_tokenized_prompt

/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package tokenizer

import (
	"fmt"
	"testing"

	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
			assert.Equal(t, tc.want, CompletionPrompts(tc.in))
		})
	}
}

func TestTokenizedPromptState_Sequences(t *testing.T) {
	tests := []struct {
		name string
		in   *TokenizedPromptState
		want [][]uint32
	}{
		{name: "nil", in: nil, want: nil},
		{name: "empty", in: &TokenizedPromptState{}, want: nil},
		{name: "single", in: &TokenizedPromptState{TokenIDs: []uint32{1, 2, 3}}, want: [][]uint32{{1, 2, 3}}},
		{
			name: "list",
			in:   &TokenizedPromptState{TokenIDsList: [][]uint32{{1}, {2, 3}}},
			want: [][]uint32{{1}, {2, 3}},
		},
		{
			name: "list takes precedence over single",
			in:   &TokenizedPromptState{TokenIDs: []uint32{99}, TokenIDsList: [][]uint32{{1}, {2}}},
			want: [][]uint32{{1}, {2}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.in.Sequences())
		})
	}
}

func TestTokenizedPromptState_TotalTokens(t *testing.T) {
	tests := []struct {
		name string
		in   *TokenizedPromptState
		want int
	}{
		{name: "nil", in: nil, want: 0},
		{name: "single", in: &TokenizedPromptState{TokenIDs: []uint32{1, 2, 3}}, want: 3},
		{name: "list", in: &TokenizedPromptState{TokenIDsList: [][]uint32{{1, 2}, {3, 4, 5}}}, want: 5},
		{name: "list with empties", in: &TokenizedPromptState{TokenIDsList: [][]uint32{{}, {1, 2}}}, want: 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.in.TotalTokens())
		})
	}
}

func TestTokenizedPromptState_Clone(t *testing.T) {
	src := &TokenizedPromptState{
		TokenIDs:     []uint32{1, 2, 3},
		TokenIDsList: [][]uint32{{4, 5}, {6}},
	}
	dst := src.Clone().(*TokenizedPromptState)
	require.Equal(t, src, dst)

	// Mutating the clone must not affect the original.
	dst.TokenIDs[0] = 999
	dst.TokenIDsList[0][0] = 999
	assert.Equal(t, uint32(1), src.TokenIDs[0])
	assert.Equal(t, uint32(4), src.TokenIDsList[0][0])
}

// TestTokenizerScorer_MultiPrompt verifies that an OpenAI-style array prompt
// renders one sequence per element, in order, and lands in TokenIDsList
// (with TokenIDs left unset).
func TestTokenizerScorer_MultiPrompt(t *testing.T) {
	ctx := utils.NewTestContext(t)
	cycleState := scheduling.NewCycleState()

	perPrompt := map[string][]uint32{
		"alpha": {1, 2},
		"beta":  {3, 4, 5},
		"gamma": {6},
	}
	var seen []string
	tok := &mockTokenizer{
		renderFunc: func(prompt string) ([]uint32, []tokenizerTypes.Offset, error) {
			seen = append(seen, prompt)
			ids, ok := perPrompt[prompt]
			if !ok {
				return nil, nil, fmt.Errorf("unexpected prompt: %s", prompt)
			}
			return ids, nil, nil
		},
	}
	p := newTestPlugin(tok)

	request := &scheduling.LLMRequest{
		RequestId: "multi",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: scheduling.Prompt{Strings: []string{"alpha", "beta", "gamma"}},
			},
		},
	}

	p.Score(ctx, cycleState, request, testEndpoints)

	stored, err := scheduling.ReadCycleStateKey[*TokenizedPromptState](
		cycleState, TokenizedPromptStateKey)
	require.NoError(t, err)
	require.NotNil(t, stored)
	assert.Empty(t, stored.TokenIDs, "multi-prompt result must leave TokenIDs unset")
	require.Equal(t, [][]uint32{{1, 2}, {3, 4, 5}, {6}}, stored.TokenIDsList)
	assert.Equal(t, []string{"alpha", "beta", "gamma"}, seen)
	assert.Equal(t, 6, stored.TotalTokens())
	assert.Len(t, stored.Sequences(), 3)
}

// TestTokenizerScorer_MultiPromptDropsEmpty verifies that empty entries in the
// prompt array never reach the tokenizer.
func TestTokenizerScorer_MultiPromptDropsEmpty(t *testing.T) {
	ctx := utils.NewTestContext(t)
	cycleState := scheduling.NewCycleState()

	var seen []string
	tok := &mockTokenizer{
		renderFunc: func(prompt string) ([]uint32, []tokenizerTypes.Offset, error) {
			seen = append(seen, prompt)
			return []uint32{1}, nil, nil
		},
	}
	p := newTestPlugin(tok)

	request := &scheduling.LLMRequest{
		RequestId: "multi-empty",
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: scheduling.Prompt{Strings: []string{"a", "", "c"}},
			},
		},
	}
	p.Score(ctx, cycleState, request, testEndpoints)

	stored, err := scheduling.ReadCycleStateKey[*TokenizedPromptState](
		cycleState, TokenizedPromptStateKey)
	require.NoError(t, err)
	require.Equal(t, []string{"a", "c"}, seen, "empty prompts must not reach the tokenizer")
	require.Len(t, stored.TokenIDsList, 2)
}
