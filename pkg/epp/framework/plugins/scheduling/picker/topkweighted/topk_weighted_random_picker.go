/*
Copyright 2026 The llm-d Authors.

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

// Package topkweighted implements a picker that bounds the candidate pool to
// the top-K (or top-K%) by score before performing A-Res weighted-random
// sampling. Useful when the candidate fleet is large enough that the
// upstream weighted-random picker leaks meaningful traffic to the long
// low-score tail.
package topkweighted

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	scheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// PluginType is the type-name of the picker.
	PluginType = "topk-weighted-random-picker"

	// DefaultMaxNumOfEndpoints is the default for maxNumOfEndpoints
	// (matches upstream picker defaults).
	DefaultMaxNumOfEndpoints = 1

	// DefaultMinTopK is the floor on the candidate-pool size after applying
	// topK / topKPercent. Without this floor, small clusters or aggressive
	// percentages would degrade to argmax (single candidate → A-Res trivially
	// picks it). The default of 2 ensures the picker actually samples.
	DefaultMinTopK = 2
)

// Parameters configures the topk-weighted-random picker.
//
// Set at least one of TopK or TopKPercent. When both are set, the more
// restrictive one (smaller candidate pool) wins. The result is then
// floored at MinTopK and capped at the actual candidate count.
type Parameters struct {
	// MaxNumOfEndpoints is the number of endpoints to return after sampling
	// (mirrors upstream pickers). Default 1.
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints,omitempty"`

	// TopK is the absolute number of top-scoring candidates to keep before
	// A-Res sampling. Optional; mutually combinable with TopKPercent.
	TopK int `json:"topK,omitempty"`

	// TopKPercent is the percentage (1-100) of the total candidate pool to
	// keep before A-Res sampling. Optional; mutually combinable with TopK.
	// Example: 25 with 40 candidates → 10. Computed via ceil(N * P / 100)
	// so the boundary always rounds up.
	TopKPercent int `json:"topKPercent,omitempty"`

	// MinTopK is the floor on the candidate-pool size after applying the
	// topK / topKPercent filter. Defaults to 2 so the picker actually
	// samples on small clusters. Set to 1 to allow degeneration to argmax.
	MinTopK int `json:"minTopK,omitempty"`
}

func (p Parameters) validate() error {
	if p.TopK == 0 && p.TopKPercent == 0 {
		return fmt.Errorf("at least one of 'topK' or 'topKPercent' must be set")
	}
	if p.TopK < 0 {
		return fmt.Errorf("'topK' must be non-negative")
	}
	if p.TopKPercent < 0 || p.TopKPercent > 100 {
		return fmt.Errorf("'topKPercent' must be in [0, 100], got %d", p.TopKPercent)
	}
	if p.MinTopK < 0 {
		return fmt.Errorf("'minTopK' must be non-negative")
	}
	return nil
}

// effectiveK computes the candidate-pool size to keep, given the total
// number of candidates, the configured topK / topKPercent, and the floor.
func (p Parameters) effectiveK(n int) int {
	if n <= 0 {
		return 0
	}

	k := n
	if p.TopK > 0 && p.TopK < k {
		k = p.TopK
	}
	if p.TopKPercent > 0 {
		// ceil(n * pct / 100) — boundary rounds up so a 1% cut on small
		// clusters keeps at least 1 (before MinTopK kicks in).
		fromPct := (n*p.TopKPercent + 99) / 100
		if fromPct < k {
			k = fromPct
		}
	}

	floor := p.MinTopK
	if floor <= 0 {
		floor = DefaultMinTopK
	}
	if k < floor {
		k = floor
	}
	if k > n {
		k = n
	}
	return k
}

// compile-time interface assertion.
var _ scheduling.Picker = &Picker{}

// PluginFactory is the registry entry point.
func PluginFactory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := Parameters{
		MaxNumOfEndpoints: DefaultMaxNumOfEndpoints,
		MinTopK:           DefaultMinTopK,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", PluginType, err)
		}
	}
	if parameters.MaxNumOfEndpoints <= 0 {
		parameters.MaxNumOfEndpoints = DefaultMaxNumOfEndpoints
	}
	if parameters.MinTopK <= 0 {
		parameters.MinTopK = DefaultMinTopK
	}
	if err := parameters.validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration for '%s' picker: %w", PluginType, err)
	}
	return New(parameters).WithName(name), nil
}

// New constructs a Picker from validated parameters.
func New(params Parameters) *Picker {
	return &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: PluginType},
		params:    params,
		rng:       newLockedRand(),
	}
}

// Picker performs A-Res weighted-random sampling restricted to the top-K
// (or top-K%) candidates by score. See package doc for rationale.
type Picker struct {
	typedName fwkplugin.TypedName
	params    Parameters
	rng       *lockedRand
}

// TypedName implements the plugin contract.
func (p *Picker) TypedName() fwkplugin.TypedName { return p.typedName }

// WithName sets the plugin's instance name.
func (p *Picker) WithName(name string) *Picker {
	p.typedName.Name = name
	return p
}

// Pick implements scheduling.Picker. Behavior:
//  1. Sort candidates by score descending.
//  2. Restrict to the top effectiveK(N) candidates.
//  3. Run A-Res weighted-random sampling on the restricted pool.
//  4. Return the top MaxNumOfEndpoints by A-Res key.
//
// If every restricted candidate has score <= 0 we fall back to uniform random
// within the restricted pool (mirrors upstream weighted-random behavior).
func (p *Picker) Pick(ctx context.Context, _ *scheduling.CycleState, scoredEndpoints []*scheduling.ScoredEndpoint) *scheduling.ProfileRunResult {
	logger := log.FromContext(ctx).V(logutil.DEBUG)

	if len(scoredEndpoints) == 0 {
		return &scheduling.ProfileRunResult{TargetEndpoints: nil}
	}

	// 1. Sort by score descending. Take a working copy — upstream pickers
	// also mutate the input slice, but explicit copy keeps the contract
	// crisp and lets us reuse the caller's slice for the result.
	working := make([]*scheduling.ScoredEndpoint, len(scoredEndpoints))
	copy(working, scoredEndpoints)
	sort.SliceStable(working, func(i, j int) bool {
		return working[i].Score > working[j].Score
	})

	// 2. Truncate to topK.
	k := p.params.effectiveK(len(working))
	working = working[:k]

	logger.Info("topk-weighted-random selecting from restricted pool",
		"total-candidates", len(scoredEndpoints),
		"effective-k", k,
		"max-num-of-endpoints", p.params.MaxNumOfEndpoints,
	)

	// 3. Detect all-zero pool — fall back to uniform-random shuffle within
	// the restricted set (preserves the topK gate).
	allZero := true
	for _, ep := range working {
		if ep.Score > 0 {
			allZero = false
			break
		}
	}
	if allZero {
		logger.Info("all top-K scores are zero, falling back to uniform random within the top-K pool")
		p.rng.Shuffle(len(working), func(i, j int) {
			working[i], working[j] = working[j], working[i]
		})
		out := working
		if p.params.MaxNumOfEndpoints < len(out) {
			out = out[:p.params.MaxNumOfEndpoints]
		}
		return resultFrom(out)
	}

	// 4. A-Res: keyᵢ = Uᵢ^(1/wᵢ); pick the largest keys.
	keys := make([]float64, len(working))
	for i, ep := range working {
		if ep.Score <= 0 {
			keys[i] = 0
			continue
		}
		u := p.rng.Float64()
		if u == 0 {
			u = 1e-10
		}
		keys[i] = math.Pow(u, 1.0/ep.Score)
	}

	idx := make([]int, len(working))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool {
		return keys[idx[i]] > keys[idx[j]]
	})

	pick := p.params.MaxNumOfEndpoints
	if pick > len(idx) {
		pick = len(idx)
	}
	out := make([]*scheduling.ScoredEndpoint, pick)
	for i := 0; i < pick; i++ {
		out[i] = working[idx[i]]
	}
	return resultFrom(out)
}

func resultFrom(eps []*scheduling.ScoredEndpoint) *scheduling.ProfileRunResult {
	targets := make([]scheduling.Endpoint, len(eps))
	for i, ep := range eps {
		targets[i] = ep
	}
	return &scheduling.ProfileRunResult{TargetEndpoints: targets}
}

// --- locked RNG ---

type lockedRand struct {
	mu sync.Mutex
	r  *rand.Rand
}

func newLockedRand() *lockedRand {
	seed := uint64(time.Now().UnixNano())
	return &lockedRand{r: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))}
}

func (r *lockedRand) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.r.Float64()
}

func (r *lockedRand) Shuffle(n int, swap func(i, j int)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.r.Shuffle(n, swap)
}
