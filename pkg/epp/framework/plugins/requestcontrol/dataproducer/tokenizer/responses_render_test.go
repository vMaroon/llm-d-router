package tokenizer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
)

// A /v1/responses request renders through vLLM's Responses render endpoint with
// the forwarded body, so its keys match what the engine computes when serving it.
func TestResponsesRenderUsesForwardedBody(t *testing.T) {
	const raw = `{"model":"client-model","instructions":"You are a coding agent.","input":[{"role":"user","content":[{"type":"input_text","text":"list files"}]},{"type":"function_call","call_id":"c1","name":"run_shell","arguments":"{\"cmd\":\"ls\"}"},{"type":"function_call_output","call_id":"c1","output":"a.go"}],"tools":[{"type":"function","name":"run_shell","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}],"reasoning":{"effort":"high"},"store":false,"stream":true}`
	var got map[string]any
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		if r.URL.Path == responsesRenderPath {
			path = r.URL.Path
			require.NoError(t, json.Unmarshal(data, &got))
		}
		_, _ = io.WriteString(w, `{"token_ids":[4,5,6]}`)
	}))
	defer srv.Close()
	renderer, err := newVLLMHTTPRenderer(&vllmConfig{URL: srv.URL}, "m")
	require.NoError(t, err)
	backend := renderBackend{tk: renderer}
	parsed, err := openai.NewOpenAIParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/responses"})
	require.NoError(t, err)
	require.NotNil(t, parsed.Body.Responses)
	before, err := parsed.Body.Payload.(fwkrh.Marshaler).Marshal()
	require.NoError(t, err)

	tp, err := backend.produce(context.Background(), parsed.Body)
	require.NoError(t, err)
	require.Equal(t, responsesRenderPath, path)
	require.Equal(t, [][]uint32{{4, 5, 6}}, tp.PerPromptTokens)
	var want map[string]any
	require.NoError(t, json.Unmarshal([]byte(raw), &want))
	want["model"] = "m"
	require.Equal(t, want, got, "renderer must receive the forwarded body with the served model name")
	after, err := parsed.Body.Payload.(fwkrh.Marshaler).Marshal()
	require.NoError(t, err)
	require.Equal(t, string(before), string(after), "rendering must not mutate forwarding")
}

func TestResponsesRenderSkipsStoredContinuations(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{"token_ids":[1]}`)
	}))
	defer srv.Close()
	renderer, err := newVLLMHTTPRenderer(&vllmConfig{URL: srv.URL}, "m")
	require.NoError(t, err)
	_, _, err = renderer.RenderResponses(context.Background(), fwkrh.PayloadMap{"input": "next", "previous_response_id": "resp_1"})
	require.ErrorContains(t, err, "stored response")
	_, _, err = renderer.RenderResponses(context.Background(), nil)
	require.ErrorContains(t, err, "parsed PayloadMap")
	require.Zero(t, calls)
}
