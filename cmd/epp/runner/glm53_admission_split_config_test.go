package runner

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	"k8s.io/apimachinery/pkg/types"
)

// TestGLM53AdmissionSplitConfigLoad drives a routing config file (GLM53_CONFIG) through both config phases.
func TestGLM53AdmissionSplitConfigLoad(t *testing.T) {
	path := os.Getenv("GLM53_CONFIG")
	if path == "" {
		t.Skip("GLM53_CONFIG unset")
	}
	opts := runserver.NewOptions()
	opts.ConfigFile = path
	opts.PoolName = "glm53-cpu04-0916"
	opts.PoolNamespace = "glm53-serving"
	opts.RefreshMetricsInterval = 80 * time.Millisecond

	r := NewRunner()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
	require.NoError(t, err)
	ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
	cfg, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds)
	require.NoError(t, err)

	t.Logf("saturation detector: %s", cfg.SaturationDetector.TypedName())
	_, tracks := cfg.SaturationDetector.(flowcontrol.TokenDispatchReservationTracker)
	require.True(t, tracks, "configured saturation detector must track dispatch reservations")
	t.Logf("flow control enabled: %v", cfg.FlowControlConfig != nil)
	t.Logf("scheduler: %s", cfg.SchedulerConfig.String())
	for _, p := range r.PluginHandle.GetAllPlugins() {
		t.Logf("plugin %s", p.TypedName())
	}

	if os.Getenv("GLM53_STAGE_CHECK") == "" {
		return
	}
	key := attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName("inflight-load-producer")
	ep := func(name, role string, waiting int, kv float64, inflight int64) fwkdl.Endpoint {
		m := fwkdl.NewMetrics()
		m.WaitingQueueSize, m.KVCacheUsagePercent, m.UpdateTime = waiting, kv, time.Now()
		e := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name, Namespace: "glm53-serving"},
			Labels: map[string]string{"llm-d.ai/role": role}}, m)
		e.GetAttributes().Put(key, &attrconcurrency.InFlightLoad{Requests: inflight})
		return e
	}
	// Prefill: waiting 3 and 1 (mean waiting/2 = 1.0), low KV, large router in-flight counts that must not matter.
	prefill := []fwkdl.Endpoint{ep("p0", "prefill", 3, 0.3, 50), ep("p1", "prefill", 1, 0.3, 50)}
	// Decode: in-flight 1 and 3 (4 / (2*2) = 1.0), deep queue and hot KV that must not matter.
	decode := []fwkdl.Endpoint{ep("d0", "decode", 7, 0.95, 1), ep("d1", "decode", 7, 0.95, 3)}
	sd := cfg.SaturationDetector
	require.InDelta(t, 1.0, sd.Saturation(flowcontrol.WithSaturationStage(ctx, "prefill"), prefill), 1e-9, "prefill stage = engine waiting only")
	require.InDelta(t, 1.0, sd.Saturation(flowcontrol.WithSaturationStage(ctx, "decode"), decode), 1e-9, "decode stage = request count only")
	decode[0].GetAttributes().Put(key, &attrconcurrency.InFlightLoad{Requests: 0})
	require.InDelta(t, 0.75, sd.Saturation(flowcontrol.WithSaturationStage(ctx, "decode"), decode), 1e-9)
	// Prefill KV is the one foreign term left in the prefill stage (kvCacheUtilThreshold 1.0 keeps it as raw usage).
	prefill[1].GetMetrics().KVCacheUsagePercent = 0.9
	t.Logf("prefill stage with one prefill rank at KV 0.9: %.3f", sd.Saturation(flowcontrol.WithSaturationStage(ctx, "prefill"), prefill))
	prefill[0].GetMetrics().UpdateTime = time.Now().Add(-time.Second)
	t.Logf("prefill stage with p0 metrics 1 s old: %.3f", sd.Saturation(flowcontrol.WithSaturationStage(ctx, "prefill"), prefill))
}
