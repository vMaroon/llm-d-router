package preciseprefixcache

import (
	"context"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// endpointToKey is a function type that converts a Pod to a string key.
// It returns the key and a boolean indicating success.
type endpointToKeyFunc func(endpoint scheduling.Endpoint) (string, bool)

// absoluteScoredPods produces per-endpoint scores in [0, 1] using
// `score / totalBlocks`, mirroring the approximate scorer's
// `matched_blocks / total_blocks` semantics. Pods missing from the score map
// (no hits) get 0.0; cold clusters (totalBlocks == 0) score 0.0 for everyone.
//
// Replaces the old min-max normalization. The change is intentional:
//   - cold cluster used to return 1.0 for every pod (full-strength noise)
//   - small absolute hits used to be stretched to 1.0 if best-in-set
//
// Tier weights (e.g. GPU=1.0, CPU=0.8) flow through unchanged because the
// upstream `score` already incorporates them; the cap below saturates the
// rare case where weighted hits could exceed totalBlocks.
func absoluteScoredPods(endpoints []scheduling.Endpoint, endpointToKey endpointToKeyFunc,
	scores map[string]float64, totalBlocks int) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))
	if totalBlocks <= 0 {
		for _, endpoint := range endpoints {
			scoredEndpoints[endpoint] = 0.0
		}
		return scoredEndpoints
	}

	denom := float64(totalBlocks)
	for _, endpoint := range endpoints {
		key, ok := endpointToKey(endpoint)
		if !ok {
			continue
		}
		raw, ok := scores[key]
		if !ok || raw <= 0 {
			scoredEndpoints[endpoint] = 0.0
			continue
		}
		ratio := raw / denom
		if ratio > 1.0 {
			ratio = 1.0
		}
		scoredEndpoints[endpoint] = ratio
	}
	return scoredEndpoints
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
