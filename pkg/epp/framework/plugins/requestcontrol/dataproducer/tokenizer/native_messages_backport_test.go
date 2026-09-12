package tokenizer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/stretchr/testify/require"
)

func TestNativeMessagesMatchesProductionForwardedBody(t *testing.T) {
	const raw = `{"model":"m","max_tokens":7,"system":"system","messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"thinking","thinking":"reason","signature":"sig"},{"type":"tool_use","id":"call1","name":"f","input":{"z":1,"a":2}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call1","content":"result"}]}],"tools":[{"name":"f","input_schema":{"properties":{"z":{"type":"number"},"a":{"type":"string"}},"type":"object"}}],"output_config":{"effort":"high"},"chat_template_kwargs":{"enable_thinking":false},"extension":{"z":9007199254740993,"a":1e0}}`
	var got []byte
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		if r.URL.Path == "/v1/messages/render" || r.URL.Path == chatRenderPath {
			var m map[string]any
			_ = json.Unmarshal(data, &m)
			if _, exists := m["tools"]; exists {
				got, path = data, r.URL.Path
			}
		}
		_, _ = io.WriteString(w, `{"token_ids":[1,2,3]}`)
	}))
	defer srv.Close()
	var cfg tokenizerPluginConfig
	require.NoError(t, json.Unmarshal([]byte(`{"modelName":"m","vllm":{"url":"`+srv.URL+`","messagesRenderMode":"native"}}`), &cfg))
	// Construct the backend directly to avoid concurrent background warmup.
	renderer, err := newVLLMHTTPRenderer(cfg.VLLM, cfg.ModelName)
	require.NoError(t, err)
	backend := renderBackend{tk: renderer}
	parsed, err := anthropic.NewAnthropicParser().ParseRequest(context.Background(), []byte(raw), map[string]string{":path": "/v1/messages"})
	require.NoError(t, err)
	want, err := parsed.Body.Payload.(fwkrh.Marshaler).Marshal()
	require.NoError(t, err)
	req := &scheduling.InferenceRequest{Body: parsed.Body}
	tp, err := backend.produce(context.Background(), req.Body)
	require.NoError(t, err)
	require.Equal(t, "/v1/messages/render", path)
	require.Equal(t, string(want), string(got), "renderer must receive the exact production-forwarded body, including effort and tool history")
	require.Equal(t, [][]uint32{{1, 2, 3}}, tp.PerPromptTokens)
	forwarded, err := req.Body.Payload.(fwkrh.Marshaler).Marshal()
	require.NoError(t, err)
	require.Equal(t, string(want), string(forwarded), "rendering must not mutate forwarding")
}

func TestNativeMessagesModeAndErrors(t *testing.T) {
	for _, mode := range []string{"", "legacy", "native"} {
		r, err := newVLLMHTTPRenderer(&vllmConfig{MessagesRenderMode: mode}, "m")
		require.NoError(t, err)
		require.Equal(t, mode == "native", r.nativeMessages)
	}
	_, err := newVLLMHTTPRenderer(&vllmConfig{MessagesRenderMode: "typo"}, "m")
	require.ErrorContains(t, err, "invalid messagesRenderMode")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	r, err := newVLLMHTTPRenderer(&vllmConfig{URL: srv.URL, MessagesRenderMode: "native"}, "m")
	require.NoError(t, err)
	_, _, err = r.RenderMessages(context.Background(), nil)
	require.ErrorContains(t, err, "parsed PayloadMap")
	_, _, err = r.RenderMessages(context.Background(), fwkrh.RawPayload(`{}`))
	require.ErrorContains(t, err, "parsed PayloadMap")
	_, _, err = r.RenderMessages(context.Background(), fwkrh.PayloadMap{"messages": []any{}})
	require.ErrorContains(t, err, "503")
}
