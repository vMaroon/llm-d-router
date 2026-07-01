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

package sessioncoverage

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid/constants"
)

const testEpsilon = 1e-9

type fakeClock struct {
	t time.Time
}

func (c *fakeClock) now() time.Time { return c.t }

func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newTestScorer(params parameters) (*SessionCoverage, *fakeClock) {
	clock := &fakeClock{t: time.Unix(1_000_000, 0)}
	//nolint:staticcheck // nil context intentionally skips the background sweeper in tests.
	s := New(nil, "test", params)
	s.now = clock.now
	return s, clock
}

func newEndpoint(name string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Namespace: "ns", Name: name}},
		&fwkdl.Metrics{},
		nil,
	)
}

func podKey(endpoint fwksched.Endpoint) string {
	return endpoint.GetMetadata().NamespacedName.String()
}

// chatRequest builds a chat-completions request whose flattened content is
// `chars` characters long, carrying the session id in the default header.
func chatRequest(sid string, chars int) *fwksched.InferenceRequest {
	headers := map[string]string{}
	if sid != "" {
		headers[defaultHeaderName] = sid
	}
	return &fwksched.InferenceRequest{
		Headers: headers,
		Body: &fwkrh.InferenceRequestBody{
			ChatCompletions: &fwkrh.ChatCompletionsRequest{
				Messages: []fwkrh.Message{
					{Role: "user", Content: fwkrh.Content{Raw: strings.Repeat("a", chars)}},
				},
			},
		},
	}
}

func schedulingResultFor(endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func endOfStreamResponse(promptTokens, completionTokens int) *requestcontrol.Response {
	return &requestcontrol.Response{
		EndOfStream: true,
		Usage: fwkrh.Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
		},
	}
}

func expectScore(t *testing.T, scores map[fwksched.Endpoint]float64, endpoint fwksched.Endpoint, want float64) {
	t.Helper()
	got, ok := scores[endpoint]
	if !ok {
		t.Fatalf("endpoint %s missing from scores", podKey(endpoint))
	}
	if math.Abs(got-want) > testEpsilon {
		t.Errorf("endpoint %s score = %v, want %v", podKey(endpoint), got, want)
	}
}

func TestScoreNoSessionID(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")

	scores := s.Score(context.Background(), chatRequest("", 4000), []fwksched.Endpoint{podA})

	expectScore(t, scores, podA, 0.0)
}

func TestScoreUnknownSession(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")

	scores := s.Score(context.Background(), chatRequest("s1", 4000), []fwksched.Endpoint{podA})

	expectScore(t, scores, podA, 0.0)
}

func TestPreRequestThenScore(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA, podB := newEndpoint("pod-a"), newEndpoint("pod-b")

	// Turn 1: 4000 chars => 1001 estimated tokens, scheduled to pod-a.
	s.PreRequest(context.Background(), chatRequest("s1", 4000), schedulingResultFor(podA))

	// Turn 2: 4800 chars => 1201 estimated tokens.
	scores := s.Score(context.Background(), chatRequest("s1", 4800), []fwksched.Endpoint{podA, podB})

	expectScore(t, scores, podA, 1001.0/1201.0)
	expectScore(t, scores, podB, 0.0)
}

func TestResponseBodyOverridesEstimate(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")
	request := chatRequest("s1", 4000)

	s.PreRequest(context.Background(), request, schedulingResultFor(podA))
	s.ResponseBody(context.Background(), request, endOfStreamResponse(1100, 100), podA.GetMetadata())

	// Coverage is now 1200 ground-truth tokens.
	scores := s.Score(context.Background(), chatRequest("s1", 5200), []fwksched.Endpoint{podA}) // 1301 tokens
	expectScore(t, scores, podA, 1200.0/1301.0)

	// A smaller, stale usage report must not shrink coverage.
	s.ResponseBody(context.Background(), request, endOfStreamResponse(400, 100), podA.GetMetadata())
	scores = s.Score(context.Background(), chatRequest("s1", 5200), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 1200.0/1301.0)
}

func TestScoreClampsToFullCoverage(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")
	request := chatRequest("s1", 4000)

	s.ResponseBody(context.Background(), request, endOfStreamResponse(1800, 200), podA.GetMetadata())

	// 4000 chars => 1001 tokens; coverage 2000 > X, at the rollover boundary
	// (1001 >= 0.5*2000) so no reset, and the fraction clamps to 1.0.
	scores := s.Score(context.Background(), chatRequest("s1", 4000), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 1.0)
}

func TestRolloverResetsSession(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")
	request := chatRequest("s1", 40000)

	s.ResponseBody(context.Background(), request, endOfStreamResponse(9500, 500), podA.GetMetadata())

	// Prompt shrinks to ~2001 tokens < 0.5 * 10000: rollover.
	scores := s.Score(context.Background(), chatRequest("s1", 8000), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 0.0)

	s.mu.Lock()
	_, exists := s.sessions["s1"]
	s.mu.Unlock()
	if exists {
		t.Error("session entry should be removed after rollover")
	}
}

func TestRolloverDisabled(t *testing.T) {
	s, _ := newTestScorer(parameters{RolloverRatio: -1})
	podA := newEndpoint("pod-a")
	request := chatRequest("s1", 40000)

	s.ResponseBody(context.Background(), request, endOfStreamResponse(9500, 500), podA.GetMetadata())

	// Same shrunken prompt as the rollover test, but detection is disabled:
	// full coverage of the small prompt.
	scores := s.Score(context.Background(), chatRequest("s1", 8000), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 1.0)
}

func TestStreamingChunksIgnoredUntilEndOfStream(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")
	request := chatRequest("s1", 4000)

	chunk := &requestcontrol.Response{EndOfStream: false, Usage: fwkrh.Usage{PromptTokens: 5000}}
	s.ResponseBody(context.Background(), request, chunk, podA.GetMetadata())

	scores := s.Score(context.Background(), chatRequest("s1", 4000), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 0.0)
}

func TestZeroUsageKeepsEstimate(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")
	request := chatRequest("s1", 4000) // 1001 estimated tokens

	s.PreRequest(context.Background(), request, schedulingResultFor(podA))
	// Stream finished without usage (no include_usage): estimate survives.
	s.ResponseBody(context.Background(), request, endOfStreamResponse(0, 0), podA.GetMetadata())

	scores := s.Score(context.Background(), chatRequest("s1", 4800), []fwksched.Endpoint{podA}) // 1201 tokens
	expectScore(t, scores, podA, 1001.0/1201.0)
}

func TestSessionsExpire(t *testing.T) {
	s, clock := newTestScorer(parameters{SessionTTLSeconds: 60})
	podA := newEndpoint("pod-a")

	s.PreRequest(context.Background(), chatRequest("s1", 4000), schedulingResultFor(podA))

	clock.advance(2 * time.Minute)
	s.removeExpired()

	scores := s.Score(context.Background(), chatRequest("s1", 4000), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 0.0)
}

func TestScoreRefreshesLastSeen(t *testing.T) {
	s, clock := newTestScorer(parameters{SessionTTLSeconds: 60})
	podA := newEndpoint("pod-a")

	s.PreRequest(context.Background(), chatRequest("s1", 4000), schedulingResultFor(podA))

	// Scoring within the TTL refreshes last-seen, so the entry survives a
	// sweep that runs just past the original insertion time.
	clock.advance(45 * time.Second)
	s.Score(context.Background(), chatRequest("s1", 4000), []fwksched.Endpoint{podA})
	clock.advance(45 * time.Second)
	s.removeExpired()

	scores := s.Score(context.Background(), chatRequest("s1", 4000), []fwksched.Endpoint{podA})
	expectScore(t, scores, podA, 1.0)
}

func TestCapacityEviction(t *testing.T) {
	s, _ := newTestScorer(parameters{MaxSessions: 2})
	podA := newEndpoint("pod-a")

	for _, sid := range []string{"s1", "s2", "s3"} {
		s.PreRequest(context.Background(), chatRequest(sid, 4000), schedulingResultFor(podA))
	}

	s.mu.Lock()
	size := len(s.sessions)
	s.mu.Unlock()
	if size > 2 {
		t.Errorf("session index size = %d, want <= 2", size)
	}
}

func TestSessionIDFromAttributePrecedesHeader(t *testing.T) {
	s, _ := newTestScorer(parameters{})
	podA := newEndpoint("pod-a")

	request := chatRequest("header-session", 4000)
	key := attrsession.SessionIDDataKey.WithNonEmptyProducerName(sessionidconstants.SessionIDProducerType).String()
	request.PutAttribute(key, attrsession.SessionID("attr-session"))

	s.PreRequest(context.Background(), request, schedulingResultFor(podA))

	s.mu.Lock()
	_, attrExists := s.sessions["attr-session"]
	_, headerExists := s.sessions["header-session"]
	s.mu.Unlock()
	if !attrExists || headerExists {
		t.Errorf("attribute session id should take precedence: attr=%v header=%v", attrExists, headerExists)
	}
}

func TestEstimatePromptTokens(t *testing.T) {
	s, _ := newTestScorer(parameters{})

	chat := chatRequest("s1", 4000)
	if got := s.estimatePromptTokens(chat); got != 1001 {
		t.Errorf("chat estimate = %d, want 1001", got)
	}

	completions := &fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: strings.Repeat("b", 2000)}},
		},
	}
	if got := s.estimatePromptTokens(completions); got != 501 {
		t.Errorf("completions estimate = %d, want 501", got)
	}

	sizeOnly := &fwksched.InferenceRequest{RequestSizeBytes: 4096}
	if got := s.estimatePromptTokens(sizeOnly); got != 1024 {
		t.Errorf("size-based estimate = %d, want 1024", got)
	}

	if got := s.estimatePromptTokens(&fwksched.InferenceRequest{}); got != 0 {
		t.Errorf("empty request estimate = %d, want 0", got)
	}
}

func TestFactoryDefaults(t *testing.T) {
	p, err := Factory("session-coverage", nil, nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	s, ok := p.(*SessionCoverage)
	if !ok {
		t.Fatalf("Factory returned %T, want *SessionCoverage", p)
	}
	if s.headerName != defaultHeaderName || s.charsPerToken != defaultCharsPerToken ||
		s.sessionTTL != defaultSessionTTL || s.maxSessions != defaultMaxSessions {
		t.Errorf("Factory defaults not applied: %+v", s)
	}
	if s.TypedName().Type != SessionCoverageScorerType {
		t.Errorf("TypedName().Type = %s, want %s", s.TypedName().Type, SessionCoverageScorerType)
	}
	if s.Category() != fwksched.Affinity {
		t.Errorf("Category() = %s, want Affinity", s.Category())
	}
}
