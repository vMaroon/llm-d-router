/*
Copyright 2025 The Kubernetes Authors.

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
// Package concurrency implements a synchronous saturation detector and scheduling filter for LLM
// routing. It consumes in-flight requests and tokens data from the Endpoint's AttributeMap
// to provide instantaneous backpressure and protect endpoints from sudden traffic bursts.
//
// For detailed architectural trade-offs and configuration, see the package README.
package concurrency

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/utilization"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

const (
	ConcurrencyDetectorType = "concurrency-detector"
)

// ConcurrencyDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func ConcurrencyDetectorFactory(
	name string,
	params *json.Decoder,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	var apiCfg apiConfig
	if params != nil {
		if err := params.Decode(&apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal concurrency detector config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	d := newDetector(name, *cfg, log.FromContext(handle.Context()))
	if len(apiCfg.DecodeSafety) > 0 {
		if cfg.mode != modeHybrid {
			return nil, fmt.Errorf("decodeSafety requires hybrid mode to retain pending-dispatch accounting")
		}
		guard, err := utilization.UtilizationDetectorFactory(name+"-decode-safety", json.NewDecoder(bytes.NewReader(apiCfg.DecodeSafety)), handle)
		if err != nil {
			return nil, err
		}
		d.decodeSafety = guard.(*utilization.Detector)
	}
	return d, nil
}

var (
	_ fwksched.Filter                             = &detector{}
	_ flowcontrol.SaturationDetector              = &detector{}
	_ flowcontrol.DispatchReservationTracker      = &detector{}
	_ flowcontrol.TokenDispatchReservationTracker = &detector{}
)

// detector implements a saturation detector and scheduling filter based on active request concurrency.
type detector struct {
	decodeSafety                 *utilization.Detector
	config                       config
	typedName                    fwkplugin.TypedName
	inFlightLoadDataKey          fwkplugin.DataKey
	uncachedRequestTokensDataKey fwkplugin.DataKey
	dispatchMu                   sync.Mutex
	dispatchReservations         map[string]int64
	pendingDispatches            atomic.Int64
	pendingTokens                atomic.Int64
}

// newDetector creates a new instance of the Concurrency Detector.
func newDetector(name string, cfg config, logger logr.Logger) *detector {
	typedName := fwkplugin.TypedName{
		Type: ConcurrencyDetectorType,
		Name: name,
	}

	pluginLogger := logger.WithName(typedName.String())
	pluginLogger.V(logutil.DEFAULT).Info("Creating new ConcurrencyDetector",
		"mode", cfg.mode,
		"maxConcurrency", cfg.maxConcurrency,
		"maxTokenConcurrency", cfg.maxTokenConcurrency,
		"maxTokenConcurrencyByRole", cfg.maxTokenConcurrencyByRole,
		"headroom", cfg.headroom)

	if cfg.headroom > 1.0 {
		pluginLogger.Info("Unusually high headroom configured; verify value is a fraction, not a percentage",
			"headroom", cfg.headroom,
			"effectiveBurst", fmt.Sprintf("%.0f%%", cfg.headroom*100))
	}

	return &detector{
		config:                       cfg,
		typedName:                    typedName,
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.inFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(cfg.inFlightLoadProducerName),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

func (d *detector) Consumes() fwkplugin.DataDependencies {
	required := map[fwkplugin.DataKey]any{
		d.inFlightLoadDataKey: attrconcurrency.InFlightLoad{},
	}
	if d.config.mode == modeTokens || d.config.mode == modeHybrid {
		required[d.uncachedRequestTokensDataKey] = attrconcurrency.UncachedRequestTokens{}
	}

	return fwkplugin.DataDependencies{
		Required: required,
	}
}

func (d *detector) getLoad(m datalayer.AttributeMap) *attrconcurrency.InFlightLoad {
	if val, ok := m.Get(d.inFlightLoadDataKey); ok {
		if load, ok := val.(*attrconcurrency.InFlightLoad); ok {
			return load
		}
	}

	return &attrconcurrency.InFlightLoad{}
}

func (d *detector) getIncomingTokens(m datalayer.AttributeMap) int64 {
	if val, ok := m.Get(d.uncachedRequestTokensDataKey); ok {
		if tokens, ok := val.(*attrconcurrency.UncachedRequestTokens); ok && tokens.Tokens > 0 {
			return tokens.Tokens
		}
	}

	return 0
}

func (d *detector) tokenCapacity(metadata *datalayer.EndpointMetadata) int64 {
	if metadata != nil {
		if capacity, ok := d.config.maxTokenConcurrencyByRole[metadata.Labels[bylabel.RoleLabel]]; ok {
			return capacity
		}
	}
	return d.config.maxTokenConcurrency
}

// ReserveDispatch accounts for a request in the interval after flow-control dispatch and before
// the in-flight load producer publishes it through PreRequest.
func (d *detector) ReserveDispatch(requestID string) bool {
	return d.ReserveDispatchTokens(requestID, 0)
}

func (d *detector) ReserveDispatchTokens(requestID string, inputTokens int64) bool {
	if requestID == "" {
		return false
	}
	d.dispatchMu.Lock()
	defer d.dispatchMu.Unlock()
	if d.dispatchReservations == nil {
		d.dispatchReservations = make(map[string]int64)
	}
	if _, loaded := d.dispatchReservations[requestID]; loaded {
		return false
	}
	inputTokens = max(inputTokens, 0)
	d.dispatchReservations[requestID] = inputTokens
	d.pendingDispatches.Add(1)
	d.pendingTokens.Add(inputTokens)
	return true
}

// ReleaseDispatch removes a dispatch reservation. Duplicate and unknown releases are no-ops.
func (d *detector) ReleaseDispatch(requestID string) bool {
	if requestID == "" {
		return false
	}
	d.dispatchMu.Lock()
	defer d.dispatchMu.Unlock()
	tokens, loaded := d.dispatchReservations[requestID]
	if !loaded {
		return false
	}
	delete(d.dispatchReservations, requestID)
	d.pendingDispatches.Add(-1)
	d.pendingTokens.Add(-tokens)
	return true
}

// Saturation calculates the saturation level of the pool.
//
// In "requests" and "tokens" mode it returns an aggregate signal, evaluated as:
//
//	Saturation = Total Inflight / Total Capacity.
//
// In "hybrid" mode saturation is instead evaluated per endpoint as
// max(requestRatio, tokenRatio) and averaged across endpoints. Evaluating each
// endpoint independently ensures an endpoint saturated on either dimension is
// reflected in the pool signal.
func (d *detector) Saturation(ctx context.Context, endpoints []datalayer.Endpoint) float64 {
	accounted := d.accountedSaturation(endpoints)
	if d.decodeSafety == nil {
		return accounted
	}
	decoders := make([]datalayer.Endpoint, 0, len(endpoints))
	for _, e := range endpoints {
		if e != nil && e.GetMetadata() != nil && e.GetMetadata().Labels[bylabel.RoleLabel] == "decode" {
			decoders = append(decoders, e)
		}
	}
	if len(decoders) == 0 {
		return accounted
	}
	return max(accounted, d.decodeSafety.Saturation(ctx, decoders))
}

func (d *detector) accountedSaturation(endpoints []datalayer.Endpoint) float64 {
	if len(endpoints) == 0 {
		return 1.0
	}

	var reqInflight, reqCapacity, tokInflight, tokCapacity int64
	var endpointCount int
	var hybridSatSum float64
	for _, e := range endpoints {
		if e == nil {
			continue
		}

		endpointCount++
		reqCapacity += d.config.maxConcurrency
		endpointTokenCapacity := d.tokenCapacity(e.GetMetadata())
		tokCapacity += endpointTokenCapacity

		if e.GetMetadata() == nil {
			continue
		}

		load := d.getLoad(e.GetAttributes())
		reqInflight += load.Requests
		tokInflight += load.Tokens
		hybridSatSum += max(
			ratio(load.Requests, d.config.maxConcurrency),
			ratio(load.Tokens, endpointTokenCapacity),
		)
	}
	d.dispatchMu.Lock()
	pendingDispatches, pendingTokens := d.pendingDispatches.Load(), d.pendingTokens.Load()
	d.dispatchMu.Unlock()

	switch d.config.mode {
	case modeTokens:
		return ratio(tokInflight+pendingTokens, tokCapacity)
	case modeHybrid:
		if endpointCount == 0 {
			return 1.0
		}
		return max(
			hybridSatSum/float64(endpointCount),
			ratio(reqInflight+pendingDispatches, reqCapacity),
			ratio(tokInflight+pendingTokens, tokCapacity),
		)
	default:
		return ratio(reqInflight+pendingDispatches, reqCapacity)
	}
}

// ratio computes inflight/capacity, failing closed (1.0) when capacity is zero.
func ratio(inflight, capacity int64) float64 {
	if capacity == 0 {
		return 1.0
	}
	return float64(inflight) / float64(capacity)
}

// Filter blocks traffic to specific endpoints that would exceed their safety limits.
//
// It applies a relaxed limit (Capacity * (1 + Headroom)) to allow for scheduling flexibility and burst tolerance.
// Token and hybrid modes include the incoming request's endpoint-specific uncached-token cost in the projection.
// In hybrid mode an endpoint is dropped when either its request load reaches the limit or its projected token load
// exceeds the limit.
func (d *detector) Filter(
	_ context.Context,
	request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
) []fwksched.Endpoint {
	// Pre-allocate assuming most endpoints will pass the filter to minimize allocations.
	filtered := make([]fwksched.Endpoint, 0, len(endpoints))

	reqLimit := int64(float64(d.config.maxConcurrency) * (1.0 + d.config.headroom))

	for _, e := range endpoints {
		if e == nil {
			continue
		}
		tokLimit := int64(float64(d.tokenCapacity(e.GetMetadata())) * (1.0 + d.config.headroom))
		load := d.getLoad(e)
		var incomingTokens int64
		if d.config.mode == modeTokens || d.config.mode == modeHybrid {
			incomingTokens = d.getIncomingTokens(e)
			// A byte estimate can exceed context capacity for a valid prompt.
			// Keep existing load limits, but do not reject on that estimate.
			if request != nil && request.Body != nil && request.Body.TokenizedPrompt == nil {
				incomingTokens = 0
			}
		}

		if d.admits(load, incomingTokens, reqLimit, tokLimit) {
			filtered = append(filtered, e)
		}
	}
	return filtered
}

// admits reports whether an endpoint is below its safety limit for the active mode.
func (d *detector) admits(load *attrconcurrency.InFlightLoad, incomingTokens, reqLimit, tokLimit int64) bool {
	projectedTokens := load.Tokens + incomingTokens

	switch d.config.mode {
	case modeTokens:
		return projectedTokens <= tokLimit
	case modeHybrid:
		return load.Requests < reqLimit && projectedTokens <= tokLimit
	default:
		return load.Requests < reqLimit
	}
}
