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

package scraped

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
)

var t0 = time.Date(2026, 9, 27, 12, 0, 0, 0, time.UTC)

func endpoint(name string, running, waiting int, updated time.Time) fwkdl.Endpoint {
	m := fwkdl.NewMetrics()
	m.RunningRequestsSize = running
	m.WaitingQueueSize = waiting
	m.UpdateTime = updated
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name, Namespace: "ns"}}, m)
}

func newTestDetector(t *testing.T, params string) *Detector {
	t.Helper()
	var api apiConfig
	require.NoError(t, json.NewDecoder(strings.NewReader(params)).Decode(&api))
	cfg, err := buildConfig(&api)
	require.NoError(t, err)
	d := NewDetector("scraped", *cfg, logr.Discard())
	d.now = func() time.Time { return t0 }
	return d
}

func TestSaturationCountsPoolLoadAgainstCandidateCapacity(t *testing.T) {
	d := newTestDetector(t, `{"maxConcurrency": 5}`)
	prefill := []fwkdl.Endpoint{endpoint("p0", 3, 4, t0), endpoint("p1", 1, 0, t0)}
	decode := []fwkdl.Endpoint{endpoint("d0", 2, 0, t0), endpoint("d1", 0, 1, t0)}
	pool := append(append([]fwkdl.Endpoint{}, prefill...), decode...)

	ctx := flowcontrol.WithSaturationStage(flowcontrol.WithSaturationPool(context.Background(), pool), "decode")
	// (3+4 + 1 + 2 + 1) / (5 * 2 decode endpoints)
	require.InDelta(t, 11.0/10.0, d.Saturation(ctx, decode), 1e-9)

	// Without a pool in the context only the candidates' own load counts.
	require.InDelta(t, 3.0/10.0, d.Saturation(context.Background(), decode), 1e-9)
}

func TestSaturationStaleEndpoints(t *testing.T) {
	d := newTestDetector(t, `{"maxConcurrency": 5, "metricsStalenessThreshold": "2s"}`)
	old := t0.Add(-3 * time.Second)
	prefill := []fwkdl.Endpoint{endpoint("p0", 4, 0, old)}
	decode := []fwkdl.Endpoint{endpoint("d0", 1, 0, t0), endpoint("d1", 5, 0, old)}
	pool := append(append([]fwkdl.Endpoint{}, prefill...), decode...)
	ctx := flowcontrol.WithSaturationPool(context.Background(), pool)

	// Stale endpoints keep their last load (4 + 5); the stale decode endpoint adds no capacity.
	require.InDelta(t, 10.0/5.0, d.Saturation(ctx, decode), 1e-9)

	// No fresh capacity is reported saturated.
	onlyStale := []fwkdl.Endpoint{endpoint("d1", 0, 0, old)}
	require.InDelta(t, 1.0, d.Saturation(flowcontrol.WithSaturationPool(context.Background(), onlyStale), onlyStale), 1e-9)
	require.InDelta(t, 1.0, d.Saturation(context.Background(), nil), 1e-9)
}

func TestSaturationMissingMetrics(t *testing.T) {
	d := newTestDetector(t, `{"maxConcurrency": 5}`)
	noMetrics := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "d9", Namespace: "ns"}}, nil)
	decode := []fwkdl.Endpoint{endpoint("d0", 2, 0, t0), noMetrics}
	require.InDelta(t, 2.0/5.0, d.Saturation(context.Background(), decode), 1e-9)
}

func TestOwnDispatchesCountUntilWindowExpires(t *testing.T) {
	d := newTestDetector(t, `{"maxConcurrency": 5, "ownDispatchWindow": "1s"}`)
	decode := []fwkdl.Endpoint{endpoint("d0", 0, 0, t0), endpoint("d1", 0, 0, t0)}

	require.False(t, d.ReserveDispatch("a"))
	require.False(t, d.ReserveDispatch("a"), "duplicate reservations are idempotent")
	require.False(t, d.ReserveDispatch("b"))
	require.False(t, d.ReserveDispatch(""))
	require.False(t, d.ReleaseDispatch("a"), "release does not end the window")
	require.InDelta(t, 2.0/10.0, d.Saturation(context.Background(), decode), 1e-9)

	now := t0.Add(1500 * time.Millisecond)
	d.now = func() time.Time { return now }
	fresh := []fwkdl.Endpoint{endpoint("d0", 1, 0, now), endpoint("d1", 1, 0, now)}
	require.InDelta(t, 2.0/10.0, d.Saturation(context.Background(), fresh), 1e-9)
}

func TestOwnDispatchWeight(t *testing.T) {
	d := newTestDetector(t, `{"maxConcurrency": 5, "ownDispatchWeight": 2}`)
	decode := []fwkdl.Endpoint{endpoint("d0", 1, 0, t0), endpoint("d1", 0, 0, t0)}
	require.False(t, d.ReserveDispatch("a"))
	// (1 scraped + 2 * 1 own) / 10
	require.InDelta(t, 3.0/10.0, d.Saturation(context.Background(), decode), 1e-9)
}

func TestBuildConfig(t *testing.T) {
	cfg, err := buildConfig(nil)
	require.NoError(t, err)
	require.Equal(t, Config{MaxConcurrency: 100, MetricsStalenessThreshold: 2 * time.Second, OwnDispatchWindow: time.Second, OwnDispatchWeight: 1}, *cfg)

	for _, bad := range []string{`{"maxConcurrency": 0}`, `{"metricsStalenessThreshold": "0s"}`, `{"ownDispatchWindow": "-1s"}`, `{"ownDispatchWeight": 0.5}`} {
		var api apiConfig
		require.NoError(t, json.NewDecoder(strings.NewReader(bad)).Decode(&api))
		_, err := buildConfig(&api)
		require.Error(t, err, bad)
	}
}
