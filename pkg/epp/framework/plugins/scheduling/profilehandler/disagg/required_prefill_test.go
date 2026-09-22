package disagg

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestRequiredPrefillNeverFallsBack(t *testing.T) {
	for _, raw := range []string{`{"requirePrefill":true}`, `{"requirePrefill":true,"prefillProfile":"prefill"}`, `{"requirePrefill":true,"stageOrder":"prefill-first"}`} {
		t.Run(raw, func(t *testing.T) {
			handle := plugin.NewEppHandle(context.Background(), nil, plugin.WithMetricsRecorder(prometheus.NewRegistry()))
			p, err := HandlerFactory("strict", plugin.StrictDecoder(json.RawMessage(raw)), handle)
			require.NoError(t, err)
			h := p.(*Handler)
			for _, result := range []*scheduling.ProfileRunResult{nil, {}, {TargetEndpoints: []scheduling.Endpoint{nil}}} {
				r, err := h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, map[string]*scheduling.ProfileRunResult{
					"decode": makeProfileRunResult("decoder"), "prefill": result,
				})
				require.ErrorContains(t, err, "required prefill")
				require.Nil(t, r)
			}
			r, err := h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, map[string]*scheduling.ProfileRunResult{
				"decode": makeProfileRunResult("decoder"), "prefill": makeProfileRunResult("prefiller"),
			})
			require.NoError(t, err)
			require.Contains(t, r.ProfileResults, "prefill")
			req := &scheduling.InferenceRequest{Headers: map[string]string{}}
			require.NoError(t, h.PreRequest(context.Background(), req, r))
			require.NotEmpty(t, req.Headers[routing.PrefillEndpointHeader])
		})
	}
}

func TestRequiredPrefillUsesConfiguredProfile(t *testing.T) {
	h := NewDisaggProfileHandler("d", "p", "e", nil, nil)
	h.requirePrefill = true
	r, err := h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, map[string]*scheduling.ProfileRunResult{
		"d": makeProfileRunResult("decoder"), "prefill": makeProfileRunResult("wrong-profile"),
	})
	require.ErrorContains(t, err, "required prefill")
	require.Nil(t, r)
	r, err = h.ProcessResults(context.Background(), &scheduling.InferenceRequest{}, map[string]*scheduling.ProfileRunResult{
		"d": makeProfileRunResult("decoder"), "p": makeProfileRunResult("prefiller"),
	})
	require.NoError(t, err)
	require.Contains(t, r.ProfileResults, "p")
}
