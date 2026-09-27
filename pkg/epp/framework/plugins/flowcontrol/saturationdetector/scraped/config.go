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

package scraped

import (
	"errors"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

const (
	// defaultMaxConcurrency matches the concurrency detector's baseline.
	defaultMaxConcurrency int64 = 100
	// defaultMetricsStalenessThreshold tolerates a few missed scrapes at the usual 200ms refresh.
	defaultMetricsStalenessThreshold = 2 * time.Second
	// defaultOwnDispatchWindow covers dispatch-to-engine transit plus one scrape.
	defaultOwnDispatchWindow = time.Second
)

type apiConfig struct {
	// MaxConcurrency is the number of in-flight requests one capacity endpoint (a candidate of the
	// stage the detector is evaluated for) is sized to hold.
	//
	// Defaults to 100 if unset.
	MaxConcurrency *int64 `json:"maxConcurrency,omitempty"`
	// MetricsStalenessThreshold is how old an endpoint's metrics can be before they are stale. A
	// stale capacity endpoint contributes no capacity; every stale endpoint keeps contributing its
	// last scraped load.
	//
	// Defaults to 2s if unset.
	MetricsStalenessThreshold *metav1.Duration `json:"metricsStalenessThreshold,omitempty"`
	// OwnDispatchWindow is how long a request this router dispatched counts on top of the scraped
	// load, covering the interval before it appears in an engine scrape.
	//
	// Defaults to 1s if unset.
	OwnDispatchWindow *metav1.Duration `json:"ownDispatchWindow,omitempty"`
}

// Config is the internal, fully-validated configuration used by the detector.
type Config struct {
	MaxConcurrency            int64
	MetricsStalenessThreshold time.Duration
	OwnDispatchWindow         time.Duration
}

func buildConfig(apiCfg *apiConfig) (*Config, error) {
	var cfg apiConfig
	if apiCfg != nil {
		cfg = *apiCfg
	}
	if cfg.MaxConcurrency == nil {
		cfg.MaxConcurrency = ptr.To(defaultMaxConcurrency)
	}
	if cfg.MetricsStalenessThreshold == nil {
		cfg.MetricsStalenessThreshold = &metav1.Duration{Duration: defaultMetricsStalenessThreshold}
	}
	if cfg.OwnDispatchWindow == nil {
		cfg.OwnDispatchWindow = &metav1.Duration{Duration: defaultOwnDispatchWindow}
	}

	var errs []error
	if *cfg.MaxConcurrency <= 0 {
		errs = append(errs, fmt.Errorf("maxConcurrency must be strictly positive, got %d", *cfg.MaxConcurrency))
	}
	if cfg.MetricsStalenessThreshold.Duration <= 0 {
		errs = append(errs, fmt.Errorf("metricsStalenessThreshold must be strictly positive, got %v",
			cfg.MetricsStalenessThreshold.Duration))
	}
	if cfg.OwnDispatchWindow.Duration < 0 {
		errs = append(errs, fmt.Errorf("ownDispatchWindow must be non-negative, got %v", cfg.OwnDispatchWindow.Duration))
	}
	if err := errors.Join(errs...); err != nil {
		return nil, fmt.Errorf("invalid scraped concurrency detector configuration: %w", err)
	}

	return &Config{
		MaxConcurrency:            *cfg.MaxConcurrency,
		MetricsStalenessThreshold: cfg.MetricsStalenessThreshold.Duration,
		OwnDispatchWindow:         cfg.OwnDispatchWindow.Duration,
	}, nil
}
