package tokenizer

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/stretchr/testify/require"
)

func TestDiscoveredModelLimits(t *testing.T) {
	var limit atomic.Int64
	limit.Store(265088)
	var bad atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/models", r.URL.Path)
		if bad.Load() {
			http.Error(w, "unavailable", 503)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "glm", "max_model_len": limit.Load()}}})
	}))
	defer server.Close()
	host, port, _ := net.SplitHostPort(server.Listener.Addr().String())
	p, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{DiscoverModelLimits: true, MinModelLen: 250000, ContextLimitLabel: "render-limit"})
	require.NoError(t, err)
	ep := discoveredEndpointWithRank("a", host, port, 0, map[string]string{"render-limit": "265088"})
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	_, err = p.Pick()
	require.Error(t, err, "unknown capacity must not be selectable")
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.NoError(t, err)
	limit.Store(240000)
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.Error(t, err, "shrunk capacity must be excluded")
	limit.Store(300000)
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.NoError(t, err)
	bad.Store(true)
	p.refreshModelLimits(context.Background(), server.Client(), "glm")
	_, err = p.Pick()
	require.Error(t, err, "failed metadata refresh must not retain an unverified target")
	p.Delete(ep.GetMetadata())
	require.Empty(t, p.capabilities)
}

func TestModelLimitLabelValidation(t *testing.T) {
	for _, v := range []string{"", "0", "-1", "not-a-number"} {
		p, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{ContextLimitLabel: "limit", MinModelLen: 200000})
		require.NoError(t, err)
		ep := discoveredEndpointWithRank("a", "127.0.0.1", "8000", 0, map[string]string{"limit": v})
		require.Error(t, p.Upsert(ep.GetMetadata()))
	}
	p, err := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{ContextLimitLabel: "limit", MinModelLen: 300000})
	require.NoError(t, err)
	ep := discoveredEndpointWithRank("a", "127.0.0.1", "8000", 0, map[string]string{"limit": "265088"})
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	_, err = p.Pick()
	require.Error(t, err)
}

func TestRenderOnlyBudgetPreservesPayload(t *testing.T) {
	for _, native := range []bool{false, true} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
			require.Equal(t, float64(1), body["max_tokens"])
			require.Equal(t, float64(1), body["max_completion_tokens"])
			require.Equal(t, float64(0), body["min_tokens"])
			require.Equal(t, []any{map[string]any{"role": "user", "content": "unchanged"}}, body["messages"])
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"token_ids":[1,2,3]}`))
		}))
		mode := "legacy"
		if native {
			mode = "native"
		}
		r, err := newVLLMHTTPRenderer(&vllmConfig{URL: server.URL, MessagesRenderMode: mode, PrefillOnly: true}, "glm")
		require.NoError(t, err)
		payload := fwkrh.PayloadMap{"max_tokens": 32000, "max_completion_tokens": 32000, "min_tokens": 5, "messages": []any{map[string]any{"role": "user", "content": "unchanged"}}}
		if native {
			_, _, err = r.RenderMessages(context.Background(), payload)
		} else {
			_, _, err = r.RenderChat(context.Background(), payload)
		}
		require.NoError(t, err)
		require.Equal(t, 32000, payload["max_tokens"])
		require.Equal(t, 5, payload["min_tokens"])
		server.Close()
	}
}

func TestModelLimitProbeDeadline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() }))
	defer server.Close()
	host, port, _ := net.SplitHostPort(server.Listener.Addr().String())
	p, _ := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{DiscoverModelLimits: true})
	require.NoError(t, p.Upsert(discoveredEndpoint("a", host, port).GetMetadata()))
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	p.refreshModelLimits(ctx, server.Client(), "glm")
	_, err := p.Pick()
	require.Error(t, err)
}

func TestRenderRejectsEmptyTokenIDs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(`{"token_ids":[]}`)) }))
	defer server.Close()
	renderer, err := newVLLMHTTPRenderer(&vllmConfig{URL: server.URL, MessagesRenderMode: "native"}, "glm")
	require.NoError(t, err)
	_, _, err = renderer.RenderMessages(context.Background(), fwkrh.PayloadMap{"messages": []any{}})
	require.Error(t, err)
	_, _, err = renderer.RenderChat(context.Background(), fwkrh.PayloadMap{"messages": []any{}})
	require.Error(t, err)
}

func TestModelLimitStaleProbeCannotResurrectDeletedEndpoint(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-release
		w.Write([]byte(`{"data":[{"id":"glm","max_model_len":300000}]}`))
	}))
	defer server.Close()
	host, port, _ := net.SplitHostPort(server.Listener.Addr().String())
	p, _ := newDiscoveredEndpointPicker(&endpointDiscoveryConfig{DiscoverModelLimits: true})
	ep := discoveredEndpoint("a", host, port)
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	done := make(chan struct{})
	go func() { p.refreshModelLimits(context.Background(), server.Client(), "glm"); close(done) }()
	<-started
	p.Delete(ep.GetMetadata())
	require.NoError(t, p.Upsert(ep.GetMetadata()))
	close(release)
	<-done
	_, err := p.Pick()
	require.Error(t, err)
}

func TestReadModelLimitRejectsWrongModelAndInvalidMetadata(t *testing.T) {
	for _, body := range []string{`{"data":[{"id":"other","max_model_len":300000}]}`, `{"data":[{"id":"glm","max_model_len":0}]}`, `{"data":[{"id":"glm","max_model_len":"300000"}]}`, "invalid"} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }))
		_, err := readModelLimit(context.Background(), server.Client(), server.URL, "glm")
		require.Error(t, err)
		server.Close()
	}
}
