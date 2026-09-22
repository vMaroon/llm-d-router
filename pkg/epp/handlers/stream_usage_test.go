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

package handlers

import (
	"fmt"
	"testing"

	"github.com/go-logr/logr"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/anthropic"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
	"github.com/stretchr/testify/require"
)

func TestContinuousUsageFrames(t *testing.T) {
	for _, fragment := range []bool{false, true} {
		t.Run(fmt.Sprint(fragment), func(t *testing.T) {
			eppmetrics.Register()
			eppmetrics.Reset()
			t.Cleanup(eppmetrics.Reset)
			s := &StreamingServer{parserRegistry: NewParserRegistry([]rh.Parser{openai.NewOpenAIParser()}, logr.Discard()), director: &mockDirector{}}
			r := &RequestContext{IncomingModelName: "stream-test", TargetModelName: "stream-test", Response: &Response{Headers: map[string]string{"content-type": "text/event-stream"}}}
			highWater := 0
			for _, n := range []int{1, 3, 2, 10} {
				data := fmt.Sprintf("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n", n, 20+n)
				if fragment {
					for i := range len(data) {
						s.HandleResponseBody(t.Context(), r, []byte(data[i:i+1]), false)
					}
				} else {
					s.HandleResponseBody(t.Context(), r, []byte(data), false)
				}
				highWater = max(highWater, n)
				require.Equal(t, highWater, r.Usage.CompletionTokens)
			}
			s.HandleResponseBody(t.Context(), r, []byte("data: [DONE]\n\n"), true)
			require.Equal(t, 20, r.Usage.PromptTokens)
			require.Equal(t, 10, r.Usage.CompletionTokens)
			h := findHistogramMetric(t, eppmetrics.InferenceObjectiveSubsystem+"_output_tokens", map[string]string{"model_name": "stream-test", "target_model_name": "stream-test"})
			require.Equal(t, uint64(1), h.GetSampleCount())
			require.Equal(t, float64(10), h.GetSampleSum())
		})
	}
}

func TestCoalescedContinuousUsage(t *testing.T) {
	s := &StreamingServer{parserRegistry: NewParserRegistry([]rh.Parser{openai.NewOpenAIParser()}, logr.Discard()), director: &mockDirector{}}
	r := &RequestContext{Response: &Response{Headers: map[string]string{"content-type": "text/event-stream"}}}
	s.HandleResponseBody(t.Context(), r, []byte("data: {\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1}}\n\ndata: {\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":10}}\n\n"), false)
	require.Equal(t, 10, r.Usage.CompletionTokens)
}

func TestAnthropicUsageRetainsPrompt(t *testing.T) {
	s := &StreamingServer{parserRegistry: NewParserRegistry([]rh.Parser{anthropic.NewAnthropicParser()}, logr.Discard()), director: &mockDirector{}}
	r := &RequestContext{Parser: anthropic.NewAnthropicParser(), Response: &Response{Headers: map[string]string{"content-type": "text/event-stream"}}}
	s.HandleResponseBody(t.Context(), r, []byte("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20}}}\n\n"), false)
	s.HandleResponseBody(t.Context(), r, []byte("event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":10}}\n\n"), false)
	require.Equal(t, 20, r.Usage.PromptTokens)
	require.Equal(t, 10, r.Usage.CompletionTokens)
}
