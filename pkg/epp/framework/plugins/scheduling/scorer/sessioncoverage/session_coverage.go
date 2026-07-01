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

// Package sessioncoverage provides a scorer that routes requests belonging to
// a session toward the endpoints that already hold that session's KV state.
//
// The scorer maintains an in-memory session index: for each session id it
// records, per endpoint, the high-water token count ("coverage") known to be
// resident on that endpoint. The index is fed from two sources:
//
//   - PreRequest: when a session request is scheduled, the destination's
//     coverage is optimistically raised to the request's estimated prompt
//     token count, so concurrent and immediately-following requests of the
//     same session observe the placement before the response completes.
//   - ResponseBody (EndOfStream): the destination's coverage is raised to
//     usage.prompt_tokens + usage.completion_tokens reported by the model
//     server, replacing the estimate with ground truth. No tokenization
//     happens in the router.
//
// Score returns, per endpoint, the fraction of the incoming request's
// estimated prompt already covered on that endpoint: min(coverage, X) / X.
// Load-aware scorers in the same scheduling profile supply the distribution
// term; this scorer is purely an affinity signal and is inert for requests
// that carry no session id.
//
// A session is a linear append chain: successive requests extend a shared
// prefix, so per-endpoint coverage is a single high-water mark. When a
// request's estimated prompt shrinks well below the session's best-known
// coverage the prefix was rewritten (e.g. history compaction); the entry is
// reset and the session re-places fresh ("rollover" detection).
package sessioncoverage

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

const (
	// SessionCoverageScorerType is the type of the SessionCoverage scorer.
	SessionCoverageScorerType = "session-coverage-scorer"

	defaultHeaderName    = "x-session-id"
	defaultCharsPerToken = 4.0
	defaultSessionTTL    = 30 * time.Minute
	defaultMaxSessions   = 100_000
	defaultRolloverRatio = 0.5

	// sweepInterval is how often expired sessions are removed in the background.
	sweepInterval = time.Minute
	// capEvictBatch bounds the number of arbitrary entries dropped when the
	// index is at capacity and none are expired (overload safety valve).
	capEvictBatch = 128
)

// parameters configures the SessionCoverage scorer.
type parameters struct {
	// HeaderName is the request header carrying the session id. Defaults to
	// x-session-id. When the session-id-producer is configured its attribute
	// takes precedence over the header.
	HeaderName string `json:"headerName"`
	// CharsPerToken is the characters-per-token ratio used to estimate prompt
	// tokens when no tokenized prompt is available. Defaults to 4.0.
	CharsPerToken float64 `json:"charsPerToken"`
	// SessionTTLSeconds is the idle time after which a session entry is
	// dropped from the index. Defaults to 1800 (30 minutes).
	SessionTTLSeconds int `json:"sessionTTLSeconds"`
	// MaxSessions caps the number of tracked sessions. Defaults to 100000.
	MaxSessions int `json:"maxSessions"`
	// RolloverRatio triggers an entry reset when the incoming prompt estimate
	// falls below this fraction of the session's best-known coverage.
	// Defaults to 0.5 when unset; a negative value disables rollover detection.
	RolloverRatio float64 `json:"rolloverRatio"`
}

// compile-time type assertions
var (
	_ fwksched.Scorer                      = &SessionCoverage{}
	_ requestcontrol.PreRequest            = &SessionCoverage{}
	_ requestcontrol.ResponseBodyProcessor = &SessionCoverage{}
)

// Factory defines the factory function for the SessionCoverage scorer.
func Factory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", SessionCoverageScorerType, err)
		}
	}

	ctx := context.Background()
	if handle != nil {
		ctx = handle.Context()
	}
	return New(ctx, name, params), nil
}

// New returns a SessionCoverage scorer. Zero-valued parameters fall back to
// their defaults. The provided context bounds the background sweeper that
// evicts idle sessions.
func New(ctx context.Context, name string, params parameters) *SessionCoverage {
	headerName := strings.ToLower(strings.TrimSpace(params.HeaderName))
	if headerName == "" {
		headerName = defaultHeaderName
	}
	charsPerToken := params.CharsPerToken
	if charsPerToken <= 0 {
		charsPerToken = defaultCharsPerToken
	}
	sessionTTL := defaultSessionTTL
	if params.SessionTTLSeconds > 0 {
		sessionTTL = time.Duration(params.SessionTTLSeconds) * time.Second
	}
	maxSessions := params.MaxSessions
	if maxSessions <= 0 {
		maxSessions = defaultMaxSessions
	}
	rolloverRatio := params.RolloverRatio
	if rolloverRatio == 0 {
		rolloverRatio = defaultRolloverRatio
	}

	s := &SessionCoverage{
		typedName:     plugin.TypedName{Type: SessionCoverageScorerType, Name: name},
		headerName:    headerName,
		charsPerToken: charsPerToken,
		sessionTTL:    sessionTTL,
		maxSessions:   maxSessions,
		rolloverRatio: rolloverRatio,
		now:           time.Now,
		sessions:      map[string]*sessionEntry{},
	}
	if ctx != nil {
		go s.sweep(ctx)
	}
	return s
}

// SessionCoverage scores endpoints by the fraction of the incoming request's
// prompt already resident on each endpoint, according to the session index.
type SessionCoverage struct {
	typedName     plugin.TypedName
	headerName    string
	charsPerToken float64
	sessionTTL    time.Duration
	maxSessions   int
	rolloverRatio float64

	now func() time.Time

	mu       sync.Mutex
	sessions map[string]*sessionEntry
}

// sessionEntry tracks one session's per-endpoint coverage high-water marks.
type sessionEntry struct {
	// coverage maps endpoint (namespaced pod name) to the highest token count
	// known resident there for this session.
	coverage map[string]int64
	lastSeen time.Time
}

func (e *sessionEntry) maxCoverage() int64 {
	var m int64
	for _, c := range e.coverage {
		if c > m {
			m = c
		}
	}
	return m
}

// TypedName returns the typed name of the plugin.
func (s *SessionCoverage) TypedName() plugin.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *SessionCoverage) Category() fwksched.ScorerCategory {
	return fwksched.Affinity
}

// Score assigns each endpoint the covered fraction of the request's estimated
// prompt, min(coverage, X)/X, and zero when the request carries no session id
// or the session is unknown.
func (s *SessionCoverage) Score(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = 0.0
	}

	sid := s.sessionID(request)
	if sid == "" {
		return scores
	}
	x := s.estimatePromptTokens(request)
	if x <= 0 {
		return scores
	}

	coverage := s.coverageFor(ctx, sid, x)
	if coverage == nil {
		return scores
	}

	for _, endpoint := range endpoints {
		c := coverage[endpoint.GetMetadata().NamespacedName.String()]
		if c <= 0 {
			continue
		}
		if c > x {
			c = x
		}
		scores[endpoint] = float64(c) / float64(x)
	}
	return scores
}

// coverageFor returns a copy of the session's coverage map, handling rollover
// detection and last-seen accounting. It returns nil for unknown sessions and
// for sessions just reset by rollover.
func (s *SessionCoverage) coverageFor(ctx context.Context, sid string, promptTokens int64) map[string]int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.sessions[sid]
	if !ok {
		return nil
	}

	if s.rolloverRatio > 0 {
		if best := entry.maxCoverage(); best > 0 && float64(promptTokens) < s.rolloverRatio*float64(best) {
			// The prompt shrank well below the session's known prefix: the
			// history was rewritten (e.g. compaction). The old KV no longer
			// prefix-matches, so drop the entry and let the session re-place.
			delete(s.sessions, sid)
			log.FromContext(ctx).V(logutil.DEFAULT).Info("session rollover detected, index entry reset",
				"scorer", s.typedName.String(), "session", sid, "promptTokens", promptTokens, "bestCoverage", best)
			return nil
		}
	}

	entry.lastSeen = s.now()
	coverage := make(map[string]int64, len(entry.coverage))
	for pod, c := range entry.coverage {
		coverage[pod] = c
	}
	return coverage
}

// PreRequest optimistically raises the scheduled endpoint's coverage to the
// request's estimated prompt tokens, so the placement is visible to the next
// request of the session before the response completes.
func (s *SessionCoverage) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	sid := s.sessionID(request)
	if sid == "" {
		return
	}
	pod := primaryTargetPod(schedulingResult)
	if pod == "" {
		return
	}
	x := s.estimatePromptTokens(request)
	if x <= 0 {
		return
	}
	s.bump(ctx, sid, pod, x)
}

// ResponseBody raises the serving endpoint's coverage to the token usage
// reported by the model server. Streaming chunks are ignored until the final
// one; responses without usage (e.g. streams without include_usage) leave the
// PreRequest estimate in place.
func (s *SessionCoverage) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest, response *requestcontrol.Response, targetEndpoint *datalayer.EndpointMetadata) {
	if response == nil || !response.EndOfStream || targetEndpoint == nil {
		return
	}
	sid := s.sessionID(request)
	if sid == "" {
		return
	}
	total := int64(response.Usage.PromptTokens) + int64(response.Usage.CompletionTokens)
	if total <= 0 {
		return
	}
	pod := targetEndpoint.NamespacedName.String()
	s.bump(ctx, sid, pod, total)

	if logger := log.FromContext(ctx); logger.V(logutil.TRACE).Enabled() {
		cached := 0
		if response.Usage.PromptTokenDetails != nil {
			cached = response.Usage.PromptTokenDetails.CachedTokens
		}
		logger.V(logutil.TRACE).Info("session coverage updated from response",
			"scorer", s.typedName.String(), "session", sid, "endpoint", pod,
			"promptTokens", response.Usage.PromptTokens, "completionTokens", response.Usage.CompletionTokens,
			"cachedTokens", cached)
	}
}

// bump raises the session's coverage high-water mark on the given endpoint.
// Coverage is monotone per endpoint; stale smaller values never overwrite.
func (s *SessionCoverage) bump(ctx context.Context, sid, pod string, tokens int64) {
	now := s.now()

	s.mu.Lock()
	defer s.mu.Unlock()

	entry, ok := s.sessions[sid]
	if !ok {
		if len(s.sessions) >= s.maxSessions {
			s.evictLocked(ctx, now)
		}
		entry = &sessionEntry{coverage: map[string]int64{}}
		s.sessions[sid] = entry
	}
	if tokens > entry.coverage[pod] {
		entry.coverage[pod] = tokens
	}
	entry.lastSeen = now
}

// evictLocked drops expired sessions and, if the index is still at capacity,
// up to capEvictBatch arbitrary entries. Callers must hold s.mu.
func (s *SessionCoverage) evictLocked(ctx context.Context, now time.Time) {
	cutoff := now.Add(-s.sessionTTL)
	for sid, entry := range s.sessions {
		if entry.lastSeen.Before(cutoff) {
			delete(s.sessions, sid)
		}
	}
	if len(s.sessions) < s.maxSessions {
		return
	}
	dropped := 0
	for sid := range s.sessions {
		delete(s.sessions, sid)
		dropped++
		if dropped >= capEvictBatch {
			break
		}
	}
	log.FromContext(ctx).V(logutil.DEFAULT).Info("session index at capacity, dropped entries",
		"scorer", s.typedName.String(), "dropped", dropped, "maxSessions", s.maxSessions)
}

// sweep periodically removes idle sessions until ctx is cancelled.
func (s *SessionCoverage) sweep(ctx context.Context) {
	ticker := time.NewTicker(sweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.removeExpired()
		}
	}
}

func (s *SessionCoverage) removeExpired() {
	cutoff := s.now().Add(-s.sessionTTL)
	s.mu.Lock()
	defer s.mu.Unlock()
	for sid, entry := range s.sessions {
		if entry.lastSeen.Before(cutoff) {
			delete(s.sessions, sid)
		}
	}
}

// sessionID resolves the request's session id: the session-id-producer
// attribute when present, otherwise the configured request header.
func (s *SessionCoverage) sessionID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	if id, ok := attrsession.ReadSessionID(request); ok && id != "" {
		return string(id)
	}
	if request.Headers == nil {
		return ""
	}
	return strings.TrimSpace(request.Headers[s.headerName])
}

// estimatePromptTokens estimates the request's prompt token count without
// tokenizing in the router: the parser-provided token count when available,
// otherwise flattened text length divided by charsPerToken, otherwise the raw
// body size. Only relative consistency across endpoints matters for scoring.
func (s *SessionCoverage) estimatePromptTokens(request *fwksched.InferenceRequest) int64 {
	if request == nil {
		return 0
	}
	if body := request.Body; body != nil {
		if n := body.TokenizedPrompt.TokenCount(); n > 0 {
			return int64(n)
		}
		chars := 0
		switch {
		case body.ChatCompletions != nil:
			for _, msg := range body.ChatCompletions.Messages {
				chars += len(msg.Content.PlainText())
			}
		case body.Completions != nil:
			chars = len(body.Completions.Prompt.PlainText())
		}
		if chars > 0 {
			return int64(float64(chars)/s.charsPerToken) + 1
		}
	}
	if request.RequestSizeBytes > 0 {
		return int64(float64(request.RequestSizeBytes) / s.charsPerToken)
	}
	return 0
}

// primaryTargetPod returns the namespaced name of the primary profile's first
// target endpoint, or "" when unavailable.
func primaryTargetPod(result *fwksched.SchedulingResult) string {
	if result == nil {
		return ""
	}
	profileResult, ok := result.ProfileResults[result.PrimaryProfileName]
	if !ok || profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
		return ""
	}
	metadata := profileResult.TargetEndpoints[0].GetMetadata()
	if metadata == nil {
		return ""
	}
	return metadata.NamespacedName.String()
}
