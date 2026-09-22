/*
Copyright 2026 The Kubernetes Authors.

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

package concurrency

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attr "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	testutils "github.com/llm-d/llm-d-router/test/utils"
	"github.com/stretchr/testify/require"
)

type measuredDecodeEndpoint struct {
	datalayer.Endpoint
	metrics *datalayer.Metrics
}

func (e *measuredDecodeEndpoint) GetMetrics() *datalayer.Metrics { return e.metrics }

func TestDecodeSafetyWithStreamingLoad(t *testing.T) {
	p, err := ConcurrencyDetectorFactory("stream-admission", json.NewDecoder(strings.NewReader(`{
		"concurrencyMode":"hybrid","maxConcurrency":8,"maxTokenConcurrency":100000,
		"decodeSafety":{"kvCacheUtilThreshold":0.9,"queueDepthThreshold":2,"metricsStalenessThreshold":"2s"}
	}`)), testutils.NewTestHandle(t.Context()))
	require.NoError(t, err)
	d := p.(*detector)
	r := newLocalRegistry()
	e := &measuredDecodeEndpoint{Endpoint: newFakeEndpoint(r, "decode"), metrics: &datalayer.Metrics{UpdateTime: time.Now(), KVCacheUsagePercent: 0.72}}
	e.GetMetadata().Labels = map[string]string{bylabel.RoleLabel: "decode"}
	r.update(e.GetMetadata().ID.String(), func(x *attr.InFlightLoad) { x.Requests = 1; x.Tokens = 20000 })
	endpoints := []datalayer.Endpoint{e}
	require.InDelta(t, 0.8, d.Saturation(t.Context(), endpoints), 0.000001, "measured KV dominates, not added to logical tokens")
	r.update(e.GetMetadata().ID.String(), func(x *attr.InFlightLoad) { x.Tokens = 95000 })
	require.InDelta(t, 0.95, d.Saturation(t.Context(), endpoints), 0.000001, "stream growth dominates")
	e.metrics.WaitingQueueSize = 3
	require.InDelta(t, 1.5, d.Saturation(t.Context(), endpoints), 0.000001, "queue pressure is visible")
	e.metrics.WaitingQueueSize = 0
	e.metrics.UpdateTime = time.Now().Add(-3 * time.Second)
	require.InDelta(t, 0.95, d.Saturation(t.Context(), endpoints), 0.000001, "stale telemetry falls back to logical accounting")
	e.metrics = nil
	require.InDelta(t, 0.95, d.Saturation(t.Context(), endpoints), 0.000001, "missing telemetry falls back to logical accounting")
	e.GetMetadata().Labels[bylabel.RoleLabel] = "prefill"
	require.InDelta(t, 0.95, d.Saturation(t.Context(), endpoints), 0.000001, "prefill admission unchanged")
}
