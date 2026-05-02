package activerequest

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

const (
	// ActiveRequestType is the type of the ActiveRequest scorer.
	ActiveRequestType = "active-request-scorer"

	// defaultRequestTimeout defines the default timeout for open requests to be
	// considered stale and removed from the cache.
	defaultRequestTimeout = 2 * time.Minute
)

// Parameters defines the parameters for the
// ActiveRequest.
type Parameters struct {
	// RequestTimeout defines the timeout for requests in seconds.
	// Once the request is "in-flight" for this duration, it is considered to
	// be timed out and dropped.
	// This field accepts duration strings like "30s", "1m", "2h".
	RequestTimeout string `json:"requestTimeout"`

	// IdleThreshold defines the maximum number of active requests for a pod
	// to be considered "idle". Pods with request count <= idleThreshold
	// will receive a score of 1.0.
	// Default: 0 (only pods with zero requests are considered idle)
	IdleThreshold int `json:"idleThreshold"`

	// MaxBusyScore defines the maximum score that can be assigned to busy pods
	// (pods with request count > idleThreshold). This creates a scoring gap
	// between idle and busy pods.
	// Range: 0.0 to 1.0
	// Default: 1.0 (no gap, current behavior)
	// Example: 0.5 means idle pods get 1.0, busiest pod gets 0.0, least busy gets 0.5
	MaxBusyScore float64 `json:"maxBusyScore"`
}

// requestEntry represents a single request in the cache. Count is the number
// of in-flight units this request contributes to each PodName — 1 for normal
// requests, len(Prompt.Strings) for OpenAI-style multi-prompt completions.
// Stored alongside PodNames so increment and decrement always agree even if
// config changes mid-request.
type requestEntry struct {
	PodNames  []string
	RequestID string
	Count     int
}

// String returns a string representation of the request entry.
func (r requestEntry) String() string {
	return fmt.Sprintf("%s:%d:%s", r.RequestID, r.Count, strings.Join(r.PodNames, "."))
}

// promptCount returns how many in-flight units a request contributes per
// endpoint. Multi-prompt completions (Prompt.Strings) count as N — the
// chosen pod processes each prompt independently. Single-prompt completions,
// chat completions, and unsupported request shapes count as 1.
func promptCount(request *scheduling.LLMRequest) int {
	if request == nil || request.Body == nil {
		return 1
	}
	if request.Body.Completions == nil {
		return 1
	}
	prompts := tokenizer.CompletionPrompts(request.Body.Completions.Prompt)
	if len(prompts) <= 1 {
		return 1
	}
	return len(prompts)
}

// endpointScores implements logr.Marshaler to lazily convert endpoint keys
// to strings only when the log line is actually written.
type endpointScores map[scheduling.Endpoint]float64

func (s endpointScores) MarshalLog() interface{} {
	result := make(map[string]float64, len(s))
	for ep, score := range s {
		result[ep.GetMetadata().NamespacedName.String()] = score
	}
	return result
}

// compile-time type assertion
var _ scheduling.Scorer = &ActiveRequest{}
var _ requestcontrol.PreRequest = &ActiveRequest{}
var _ requestcontrol.ResponseBody = &ActiveRequest{}

// Factory defines the factory function for the ActiveRequest scorer.
func Factory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := Parameters{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", ActiveRequestType, err)
		}
	}

	return NewActiveRequest(handle.Context(), &parameters).WithName(name), nil
}

// NewActiveRequest creates a new ActiveRequest scorer.
func NewActiveRequest(ctx context.Context, params *Parameters) *ActiveRequest {
	requestTimeout := defaultRequestTimeout
	logger := log.FromContext(ctx)

	if params != nil && params.RequestTimeout != "" {
		paramsRequestTimeout, err := time.ParseDuration(params.RequestTimeout)
		if err != nil || paramsRequestTimeout <= 0 {
			logger.Error(err, "Invalid request timeout duration, using default request timeout")
		} else {
			requestTimeout = paramsRequestTimeout
			logger.Info("Using request timeout", "requestTimeout", requestTimeout)
		}
	}

	// cache for individual requests with their own TTL
	requestCache := ttlcache.New[string, *requestEntry](
		ttlcache.WithTTL[string, *requestEntry](requestTimeout),
		ttlcache.WithDisableTouchOnHit[string, *requestEntry](),
	)

	// Set idle threshold (default: 0)
	idleThreshold := 0
	if params != nil && params.IdleThreshold >= 0 {
		idleThreshold = params.IdleThreshold
	}

	// Set max busy score (default: 1.0)
	maxBusyScore := 1.0
	if params != nil && params.MaxBusyScore >= 0 && params.MaxBusyScore <= 1.0 {
		maxBusyScore = params.MaxBusyScore
	}

	if idleThreshold != 0 || maxBusyScore != 1.0 {
		logger.Info("Active request scorer configured with idle preference",
			"idleThreshold", idleThreshold,
			"maxBusyScore", maxBusyScore)
	}

	scorer := &ActiveRequest{
		typedName:      plugin.TypedName{Type: ActiveRequestType},
		requestCache:   requestCache,
		endpointCounts: make(map[string]int),
		mutex:          &sync.RWMutex{},
		idleThreshold:  idleThreshold,
		maxBusyScore:   maxBusyScore,
	}
	// callback to decrement count when requests expire
	// most requests will be removed in ResponseComplete, but this ensures
	// that we don't leak endpoint counts if ResponseComplete is not called
	requestCache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
		item *ttlcache.Item[string, *requestEntry]) {
		if reason == ttlcache.EvictionReasonExpired {
			entry := item.Value()
			delta := entry.Count
			if delta <= 0 {
				delta = 1
			}
			for _, endpointName := range entry.PodNames {
				scorer.decrementPodCount(endpointName, delta)
			}
		}
	})

	go cleanCachePeriodically(ctx, requestCache, requestTimeout)

	return scorer
}

// ActiveRequest keeps track of individual requests being served
// per endpoint.
type ActiveRequest struct {
	typedName plugin.TypedName

	// requestCache stores individual request entries with unique composite keys (endpointName.requestID)
	requestCache *ttlcache.Cache[string, *requestEntry]

	// endpointCounts maintains fast lookup for request counts per endpoint
	endpointCounts map[string]int
	mutex          *sync.RWMutex

	// idleThreshold defines the max request count to be considered idle
	idleThreshold int
	// maxBusyScore defines the maximum score for busy (non-idle) pods
	maxBusyScore float64
}

// TypedName returns the typed name of the plugin.
func (s *ActiveRequest) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *ActiveRequest) WithName(name string) *ActiveRequest {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *ActiveRequest) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Score scores the given endpoints based on the number of active requests
// being served by each endpoint. The score is normalized to a range of 0-1.
func (s *ActiveRequest) Score(ctx context.Context, _ *scheduling.CycleState, _ *scheduling.LLMRequest,
	endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[string]int)
	maxCount := 0
	s.mutex.RLock()
	for endpointName, count := range s.endpointCounts {
		scoredEndpoints[endpointName] = count
		if count >= maxCount {
			maxCount = count
		}
	}
	s.mutex.RUnlock()

	log.FromContext(ctx).V(logutil.TRACE).Info("Active request counts", "endpointCounts", scoredEndpoints, "maxCount", maxCount)

	scoredEndpointsMap := make(map[scheduling.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		endpointName := endpoint.GetMetadata().NamespacedName.String()
		count, exists := scoredEndpoints[endpointName]
		if !exists {
			// Pod not tracked = no requests = idle
			scoredEndpointsMap[endpoint] = 1.0
			continue
		}

		// Check if pod is idle (count <= idleThreshold)
		if count <= s.idleThreshold {
			scoredEndpointsMap[endpoint] = 1.0 // Idle pods always get max score
			continue
		}

		// Busy pod: scale from 0 to maxBusyScore
		scoredEndpointsMap[endpoint] = float64(maxCount-count) / float64(maxCount) * s.maxBusyScore
	}

	log.FromContext(ctx).V(logutil.TRACE).Info("Scored endpoints", "scores", endpointScores(scoredEndpointsMap))
	return scoredEndpointsMap
}

// PreRequest is called before a request is sent to the target endpoint.
// It creates a new request entry in the cache with its own TTL and
// increments each target endpoint's count by promptCount(request) so an
// OpenAI-style multi-prompt completion (Prompt.Strings) loads the chosen
// pod by N units instead of 1.
func (s *ActiveRequest) PreRequest(
	ctx context.Context,
	request *scheduling.LLMRequest,
	schedulingResult *scheduling.SchedulingResult,
) {
	traceLogger := log.FromContext(ctx).V(logutil.TRACE)

	count := promptCount(request)

	endpointNames := make([]string, 0, len(schedulingResult.ProfileResults))
	for profileName, profileResult := range schedulingResult.ProfileResults {
		if profileResult == nil || len(profileResult.TargetEndpoints) == 0 {
			continue
		}

		endpointName := profileResult.TargetEndpoints[0].GetMetadata().NamespacedName.String()
		endpointNames = append(endpointNames, endpointName)
		s.incrementPodCount(endpointName, count)
		traceLogger.Info(
			"Added request to cache",
			"requestId", request.RequestId,
			"endpointName", endpointName,
			"profileName", profileName,
			"count", count,
		)
	}

	// add to request cache
	s.requestCache.Set(request.RequestId, &requestEntry{PodNames: endpointNames, RequestID: request.RequestId, Count: count}, 0) // Use default TTL
}

// ResponseBody is called after a response is sent to the client.
// It removes the specific request entry from the cache and decrements
// the endpoint count.
func (s *ActiveRequest) ResponseBody(
	ctx context.Context,
	request *scheduling.LLMRequest,
	resp *requestcontrol.Response,
	targetPod *datalayer.EndpointMetadata,
) {
	traceLogger := log.FromContext(ctx).V(logutil.TRACE).WithName("ActiveRequest.ResponseBody")
	if !resp.EndOfStream {
		traceLogger.Info("Skipping ResponseBody because EndOfStream is false")
		return
	}
	if targetPod == nil {
		traceLogger.Info("Skipping ResponseBody because targetPod is nil")
		return
	}

	if item, found := s.requestCache.GetAndDelete(request.RequestId); found {
		entry := item.Value()
		if entry != nil {
			delta := entry.Count
			if delta <= 0 {
				delta = 1
			}
			for _, endpointName := range entry.PodNames {
				s.decrementPodCount(endpointName, delta)
			}
			traceLogger.Info("Removed request from cache", "requestEntry", entry.String())
		} else {
			traceLogger.Info("Request entry value is nil", "requestId", request.RequestId)
		}
	} else {
		traceLogger.Info("Request not found in cache", "requestId", request.RequestId)
	}
}

// incrementPodCount adds delta to the in-flight request count for a endpoint.
// Multi-prompt requests pass delta = len(Prompt.Strings); single-prompt /
// chat / unsupported pass delta = 1. Non-positive deltas are clamped to 1
// so a misconfigured caller can't silently no-op the increment.
func (s *ActiveRequest) incrementPodCount(endpointName string, delta int) {
	if delta <= 0 {
		delta = 1
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()

	s.endpointCounts[endpointName] += delta
}

// decrementPodCount subtracts delta from the in-flight count for a endpoint
// and removes the entry if the count reaches zero. The delta MUST match the
// value passed to the corresponding incrementPodCount; storing it on
// requestEntry guarantees that.
func (s *ActiveRequest) decrementPodCount(endpointName string, delta int) {
	if delta <= 0 {
		delta = 1
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if count, exists := s.endpointCounts[endpointName]; exists {
		if count <= delta {
			delete(s.endpointCounts, endpointName)
		} else {
			s.endpointCounts[endpointName] = count - delta
		}
	}
}

func cleanCachePeriodically[K comparable, V any](ctx context.Context, cache *ttlcache.Cache[K, V], requestTimeout time.Duration) {
	ticker := time.NewTicker(requestTimeout)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cache.DeleteExpired()
		}
	}
}
