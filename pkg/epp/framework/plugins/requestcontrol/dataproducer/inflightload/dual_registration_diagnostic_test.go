package inflightload

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	runtime "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/tokenload"
	utils "github.com/llm-d/llm-d-router/test/utils"
)

// This diagnostic exercises registration, accounting and scoring together.
func TestDualProducerRegistrationDiagnostic(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	handle := utils.NewTestHandle(ctx)
	a, err := InFlightLoadProducerFactory("inflight-load-producer", json.NewDecoder(strings.NewReader(`{"addEstimatedOutputTokens":false}`)), handle)
	require.NoError(t, err)
	b, err := InFlightLoadProducerFactory("decode-inflight-load", json.NewDecoder(strings.NewReader(`{"addEstimatedOutputTokens":true,"maxEstimatedOutputTokens":32768}`)), handle)
	require.NoError(t, err)
	first, second := a.(*InFlightLoadProducer), b.(*InFlightLoadProducer)
	r := runtime.NewRuntime(0)
	require.NoError(t, first.RegisterDependencies(r))
	require.NoError(t, second.RegisterDependencies(r))
	require.NoError(t, r.Configure(nil, logr.Discard()))
	var endpoints []sched.Endpoint
	for _, name := range []string{"busy", "idle"} {
		ep := r.NewEndpoint(ctx, &dl.EndpointMetadata{ID: types.NamespacedName{Namespace: "test", Name: name}})
		t.Cleanup(func() { r.ReleaseEndpoint(ep) })
		endpoints = append(endpoints, sched.NewEndpoint(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes()))
	}
	busy := makeTokenRequest("existing", 100000)
	result := &sched.SchedulingResult{ProfileResults: map[string]*sched.ProfileRunResult{"decode": {TargetEndpoints: endpoints[:1]}}}
	require.NoError(t, second.PreRequest(ctx, busy, result))
	require.Equal(t, int64(132768), second.GetTokens("test/busy"))
	incoming := makeTokenRequest("incoming", 1000)
	require.NoError(t, second.Produce(ctx, incoming, endpoints))
	plugin, err := tokenload.TokenLoadScorerFactory("decode-token-load", json.NewDecoder(strings.NewReader(`{"inFlightLoadProducerName":"decode-inflight-load","queueThresholdTokens":347136}`)), handle)
	require.NoError(t, err)
	scorer := plugin.(sched.Scorer)
	scores := scorer.Score(ctx, incoming, endpoints)
	_, attached := endpoints[0].Get(second.dk)
	t.Logf("tracked=%d attached=%v busy_score=%f idle_score=%f", second.GetTokens("test/busy"), attached, scores[endpoints[0]], scores[endpoints[1]])
	// Manually attaching the missing listener is a diagnostic control, not a runtime fix.
	endpoints[0].Put(second.dk, &dl.DynamicAttribute{Get: func() dl.Cloneable { return second.CrossReplicaState().Supply("test/busy")() }})
	control := scorer.Score(ctx, incoming, endpoints)
	require.Less(t, control[endpoints[0]], control[endpoints[1]])
	t.Logf("attached control: busy_score=%f idle_score=%f", control[endpoints[0]], control[endpoints[1]])
	t.Run("registered_path_must_see_load", func(t *testing.T) {
		require.Less(t, scores[endpoints[0]], scores[endpoints[1]], "tracked busy decoder must not tie with idle decoder")
	})
	t.Run("attached_attribute_control", func(t *testing.T) {
		require.Less(t, control[endpoints[0]], control[endpoints[1]])
	})
}
