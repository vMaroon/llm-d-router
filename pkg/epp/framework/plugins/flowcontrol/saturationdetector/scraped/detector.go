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

// Package scraped implements a saturation detector that counts in-flight requests from engine
// telemetry instead of router-local accounting, so every router replica in front of the same
// pool computes the same signal.
//
// For semantics and configuration, see the package README.
package scraped

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// ScrapedConcurrencyDetectorType is the unique identifier for this plugin.
	ScrapedConcurrencyDetectorType = "scraped-concurrency-detector"

	// staleWarnInterval bounds how often stale telemetry is logged; Saturation runs every
	// dispatch cycle.
	staleWarnInterval = 30 * time.Second
)

var (
	_ flowcontrol.SaturationDetector         = &Detector{}
	_ flowcontrol.DispatchReservationTracker = &Detector{}
)

// ScrapedConcurrencyDetectorFactory instantiates the detector from its JSON parameters.
func ScrapedConcurrencyDetectorFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var apiCfg apiConfig
	if params != nil {
		if err := params.Decode(&apiCfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal scraped concurrency detector config: %w", err)
		}
	}
	cfg, err := buildConfig(&apiCfg)
	if err != nil {
		return nil, err
	}
	return NewDetector(name, *cfg, log.FromContext(handle.Context())), nil
}

// Detector computes saturation as scraped in-flight requests over scraped capacity.
type Detector struct {
	config    Config
	typedName fwkplugin.TypedName
	logger    logr.Logger
	now       func() time.Time

	// ownMu guards own, the dispatch time of each request this router released recently.
	ownMu sync.Mutex
	own   map[string]time.Time

	lastStaleWarnNanos atomic.Int64
}

// NewDetector creates a scraped concurrency detector.
func NewDetector(name string, cfg Config, logger logr.Logger) *Detector {
	typedName := fwkplugin.TypedName{Type: ScrapedConcurrencyDetectorType, Name: name}
	pluginLogger := logger.WithName(typedName.String())
	pluginLogger.V(logutil.DEFAULT).Info("Creating new ScrapedConcurrencyDetector",
		"maxConcurrency", cfg.MaxConcurrency,
		"metricsStalenessThreshold", cfg.MetricsStalenessThreshold.String(),
		"ownDispatchWindow", cfg.OwnDispatchWindow.String())
	return &Detector{
		config:    cfg,
		typedName: typedName,
		logger:    pluginLogger,
		now:       time.Now,
		own:       make(map[string]time.Time),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *Detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

// Saturation returns
//
//	(scraped running + waiting on every pool endpoint + own recent dispatches) /
//	(MaxConcurrency * candidates with fresh metrics)
//
// The candidates are the endpoints of the stage being evaluated and supply the capacity. The
// load is read from the whole pool (flowcontrol.SaturationPoolFromContext), so a request still
// in prefill already counts against decode capacity, as it does for router-local accounting from
// dispatch. Without a pool in the context, the candidates are the pool.
//
// A stale endpoint keeps contributing its last scraped load, and a stale candidate contributes
// no capacity. With no fresh candidate the pool is reported saturated.
func (d *Detector) Saturation(ctx context.Context, candidates []datalayer.Endpoint) float64 {
	now := d.now()
	pool := flowcontrol.SaturationPoolFromContext(ctx)
	if pool == nil {
		pool = candidates
	}

	var inflight int64
	stale := 0
	for _, e := range pool {
		if e == nil {
			continue
		}
		m := e.GetMetrics()
		if m == nil {
			stale++
			continue
		}
		if now.Sub(m.UpdateTime) > d.config.MetricsStalenessThreshold {
			stale++
		}
		inflight += int64(m.RunningRequestsSize + m.WaitingQueueSize)
	}

	var fresh int64
	for _, e := range candidates {
		if e == nil {
			continue
		}
		if m := e.GetMetrics(); m != nil && now.Sub(m.UpdateTime) <= d.config.MetricsStalenessThreshold {
			fresh++
		}
	}

	metrics.RecordFlowControlStaleEndpoints(d.typedName.Name, stale)
	if stale > 0 {
		d.maybeLogStale(stale, len(pool))
	}
	if fresh == 0 {
		return 1.0
	}
	inflight += int64(d.ownRecent(now))
	return float64(inflight) / float64(fresh*d.config.MaxConcurrency)
}

// ReserveDispatch records a request this router released. It counts for OwnDispatchWindow,
// until the engines' scraped load includes it. It never reports a reservation, so the framework
// has nothing to release.
func (d *Detector) ReserveDispatch(requestID string) bool {
	if requestID == "" {
		return false
	}
	d.ownMu.Lock()
	defer d.ownMu.Unlock()
	if _, ok := d.own[requestID]; !ok {
		d.own[requestID] = d.now()
	}
	return false
}

// ReleaseDispatch is a no-op: the framework releases after PreRequest, before the request
// reaches an engine, so a recorded dispatch instead expires after OwnDispatchWindow.
func (d *Detector) ReleaseDispatch(string) bool {
	return false
}

func (d *Detector) ownRecent(now time.Time) int {
	d.ownMu.Lock()
	defer d.ownMu.Unlock()
	for id, at := range d.own {
		if now.Sub(at) > d.config.OwnDispatchWindow {
			delete(d.own, id)
		}
	}
	return len(d.own)
}

func (d *Detector) maybeLogStale(stale, total int) {
	now := time.Now().UnixNano()
	last := d.lastStaleWarnNanos.Load()
	if now-last < int64(staleWarnInterval) || !d.lastStaleWarnNanos.CompareAndSwap(last, now) {
		return
	}
	d.logger.V(logutil.DEFAULT).Info(
		"Endpoints with missing or stale metrics keep their last scraped load and add no capacity",
		"staleEndpoints", stale,
		"totalEndpoints", total,
		"metricsStalenessThreshold", d.config.MetricsStalenessThreshold.String())
}
