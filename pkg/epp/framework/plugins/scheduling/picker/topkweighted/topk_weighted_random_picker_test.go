/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package topkweighted

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	scheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func makeScoredEndpoint(name string, score float64) *scheduling.ScoredEndpoint {
	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: name},
			Address:        "10.0.0." + name,
			Port:           "8080",
		}, nil, nil,
	)
	return &scheduling.ScoredEndpoint{Endpoint: ep, Score: score}
}

func TestParameters_validate(t *testing.T) {
	tests := []struct {
		name    string
		in      Parameters
		wantErr bool
	}{
		{name: "neither set", in: Parameters{}, wantErr: true},
		{name: "topK only", in: Parameters{TopK: 5}, wantErr: false},
		{name: "topKPercent only", in: Parameters{TopKPercent: 25}, wantErr: false},
		{name: "both set", in: Parameters{TopK: 5, TopKPercent: 25}, wantErr: false},
		{name: "negative topK", in: Parameters{TopK: -1, TopKPercent: 25}, wantErr: true},
		{name: "topKPercent below 0", in: Parameters{TopKPercent: -1}, wantErr: true},
		{name: "topKPercent above 100", in: Parameters{TopKPercent: 101}, wantErr: true},
		{name: "negative minTopK", in: Parameters{TopK: 5, MinTopK: -1}, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.in.validate()
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestParameters_effectiveK(t *testing.T) {
	tests := []struct {
		name string
		p    Parameters
		n    int
		want int
	}{
		// MinTopK floor (defaults to 2 when unset).
		{name: "zero candidates returns 0", p: Parameters{TopKPercent: 25, MinTopK: 2}, n: 0, want: 0},
		{name: "topKPercent 25% of 40 = 10", p: Parameters{TopKPercent: 25, MinTopK: 2}, n: 40, want: 10},
		{name: "topKPercent ceil rounds up", p: Parameters{TopKPercent: 25, MinTopK: 2}, n: 7, want: 2}, // ceil(1.75)=2 → also matches floor
		{name: "topKPercent 1% with floor 2", p: Parameters{TopKPercent: 1, MinTopK: 2}, n: 100, want: 2},
		{name: "topKPercent 100% returns N", p: Parameters{TopKPercent: 100, MinTopK: 2}, n: 40, want: 40},

		{name: "topK=5 with N=40", p: Parameters{TopK: 5, MinTopK: 2}, n: 40, want: 5},
		{name: "topK exceeds N caps at N", p: Parameters{TopK: 100, MinTopK: 2}, n: 10, want: 10},
		{name: "topK below floor lifts to floor", p: Parameters{TopK: 1, MinTopK: 3}, n: 40, want: 3},

		// Both set: most restrictive wins.
		{name: "both set: topK is tighter", p: Parameters{TopK: 3, TopKPercent: 25, MinTopK: 1}, n: 40, want: 3},
		{name: "both set: percent is tighter", p: Parameters{TopK: 50, TopKPercent: 10, MinTopK: 1}, n: 40, want: 4},

		// Floor of 1 disables degeneration guard.
		{name: "minTopK=1 allows argmax", p: Parameters{TopKPercent: 1, MinTopK: 1}, n: 100, want: 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.p.effectiveK(tc.n))
		})
	}
}

// TestPick_RestrictsToTopK runs the picker many times and asserts that no
// endpoint outside the top-K (by score) is ever selected.
func TestPick_RestrictsToTopK(t *testing.T) {
	const trials = 2000
	// 10 endpoints, scores 1..10. topKPercent=30 (ceil) → top 3.
	eps := make([]*scheduling.ScoredEndpoint, 10)
	for i := 0; i < 10; i++ {
		eps[i] = makeScoredEndpoint(fmt.Sprintf("pod-%d", i), float64(i+1))
	}

	pkr := New(Parameters{
		MaxNumOfEndpoints: 1,
		TopKPercent:       30,
		MinTopK:           2,
	})

	allowed := map[string]bool{"pod-7": true, "pod-8": true, "pod-9": true}
	hits := map[string]int{}
	for i := 0; i < trials; i++ {
		// Shuffle each call to verify Pick isn't relying on input order.
		shuffled := append([]*scheduling.ScoredEndpoint(nil), eps...)
		shuffled[0], shuffled[5] = shuffled[5], shuffled[0]
		shuffled[2], shuffled[8] = shuffled[8], shuffled[2]

		res := pkr.Pick(context.Background(), nil, shuffled)
		require.Len(t, res.TargetEndpoints, 1)
		name := res.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		hits[name]++
		assert.True(t, allowed[name], "selected endpoint %q not in top-3 by score", name)
	}
	// Every allowed pod should be picked at least once across 2000 trials
	// — this is a sanity check on the weighted-random behavior.
	for n := range allowed {
		assert.Greater(t, hits[n], 0, "expected pod %q to be sampled at least once", n)
	}
}

// TestPick_FloorAvoidsArgmax verifies the MinTopK floor: with topKPercent=1
// on 5 candidates we'd otherwise get K=1 (argmax), but MinTopK=2 keeps two
// candidates so the picker actually samples between them.
func TestPick_FloorAvoidsArgmax(t *testing.T) {
	const trials = 2000
	eps := []*scheduling.ScoredEndpoint{
		makeScoredEndpoint("low", 1),
		makeScoredEndpoint("mid", 5),
		makeScoredEndpoint("high1", 9),
		makeScoredEndpoint("high2", 10),
		makeScoredEndpoint("verylow", 0.5),
	}

	pkr := New(Parameters{
		MaxNumOfEndpoints: 1,
		TopKPercent:       1, // would be 1 candidate...
		MinTopK:           2, // ...but floored to 2
	})

	hits := map[string]int{}
	for i := 0; i < trials; i++ {
		res := pkr.Pick(context.Background(), nil, eps)
		hits[res.TargetEndpoints[0].GetMetadata().NamespacedName.Name]++
	}
	// Both "high1" and "high2" must be sampled. Other pods must never win.
	assert.Greater(t, hits["high1"], 0)
	assert.Greater(t, hits["high2"], 0)
	for _, n := range []string{"low", "mid", "verylow"} {
		assert.Equal(t, 0, hits[n], "pod %q outside top-2 must never be picked", n)
	}
	// "high2" (score 10) should win more often than "high1" (score 9) under
	// A-Res — sanity check that weights still matter inside the topK.
	assert.Greater(t, hits["high2"], hits["high1"])
}

// TestPick_AllZeroFallsBackToUniform verifies that when every top-K
// candidate has score 0, the picker uniformly samples within the restricted
// pool instead of returning the same pod every time.
func TestPick_AllZeroFallsBackToUniform(t *testing.T) {
	const trials = 2000
	eps := []*scheduling.ScoredEndpoint{
		makeScoredEndpoint("a", 0),
		makeScoredEndpoint("b", 0),
		makeScoredEndpoint("c", 0),
		makeScoredEndpoint("d", 0),
	}

	pkr := New(Parameters{MaxNumOfEndpoints: 1, TopK: 3, MinTopK: 2})

	hits := map[string]int{}
	for i := 0; i < trials; i++ {
		res := pkr.Pick(context.Background(), nil, eps)
		hits[res.TargetEndpoints[0].GetMetadata().NamespacedName.Name]++
	}
	require.GreaterOrEqual(t, len(hits), 3, "uniform sampling must hit at least 3 distinct pods over %d trials", trials)
}

// TestPick_NeverPicksZeroBelowTopK verifies that even when some top-K
// candidates have positive scores and others zero, only positive-score
// candidates within the top-K are selected.
func TestPick_NeverPicksZeroBelowTopK(t *testing.T) {
	const trials = 2000
	// pod-a, pod-b: positive (in top-2). pod-c..pod-e: 0 (excluded).
	eps := []*scheduling.ScoredEndpoint{
		makeScoredEndpoint("a", 7),
		makeScoredEndpoint("b", 3),
		makeScoredEndpoint("c", 0),
		makeScoredEndpoint("d", 0),
		makeScoredEndpoint("e", 0),
	}
	pkr := New(Parameters{MaxNumOfEndpoints: 1, TopK: 2, MinTopK: 1})

	for i := 0; i < trials; i++ {
		res := pkr.Pick(context.Background(), nil, eps)
		name := res.TargetEndpoints[0].GetMetadata().NamespacedName.Name
		require.Contains(t, []string{"a", "b"}, name, "iter %d picked %q", i, name)
	}
}

// TestPick_MaxNumOfEndpointsBounded ensures Pick respects the output count
// independently of the candidate-pool restriction.
func TestPick_MaxNumOfEndpointsBounded(t *testing.T) {
	eps := make([]*scheduling.ScoredEndpoint, 20)
	for i := 0; i < 20; i++ {
		eps[i] = makeScoredEndpoint(fmt.Sprintf("p-%d", i), float64(i+1))
	}
	pkr := New(Parameters{MaxNumOfEndpoints: 3, TopKPercent: 25, MinTopK: 2}) // top 5

	res := pkr.Pick(context.Background(), nil, eps)
	require.Len(t, res.TargetEndpoints, 3)
	// All three results must come from the top 5 by score.
	allowed := map[string]bool{"p-15": true, "p-16": true, "p-17": true, "p-18": true, "p-19": true}
	for _, ep := range res.TargetEndpoints {
		assert.True(t, allowed[ep.GetMetadata().NamespacedName.Name],
			"output endpoint %q not in top-5", ep.GetMetadata().NamespacedName.Name)
	}
}

// TestPick_EmptyInput returns no targets without panicking.
func TestPick_EmptyInput(t *testing.T) {
	pkr := New(Parameters{TopKPercent: 25, MinTopK: 2})
	res := pkr.Pick(context.Background(), nil, nil)
	assert.Empty(t, res.TargetEndpoints)
}

// TestPluginFactory_Validation exercises the JSON entry point.
func TestPluginFactory_Validation(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{name: "empty config rejected", raw: `{}`, wantErr: true}, // neither topK nor topKPercent set
		{name: "topK only ok", raw: `{"topK": 5}`, wantErr: false},
		{name: "topKPercent only ok", raw: `{"topKPercent": 25}`, wantErr: false},
		{name: "topKPercent too high", raw: `{"topKPercent": 200}`, wantErr: true},
		{name: "negative topK", raw: `{"topK": -3}`, wantErr: true},
		{name: "invalid json", raw: `{not json}`, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := PluginFactory("topk", json.RawMessage(tc.raw), nil)
			if tc.wantErr {
				assert.Error(t, err)
				assert.Nil(t, p)
			} else {
				require.NoError(t, err)
				require.NotNil(t, p)
			}
		})
	}
}

// TestPluginFactory_Defaults confirms unset MaxNumOfEndpoints / MinTopK fall
// back to documented defaults rather than 0.
func TestPluginFactory_Defaults(t *testing.T) {
	p, err := PluginFactory("topk", json.RawMessage(`{"topKPercent": 50}`), nil)
	require.NoError(t, err)
	pk, ok := p.(*Picker)
	require.True(t, ok)
	assert.Equal(t, DefaultMaxNumOfEndpoints, pk.params.MaxNumOfEndpoints)
	assert.Equal(t, DefaultMinTopK, pk.params.MinTopK)
}
