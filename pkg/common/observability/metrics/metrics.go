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

package metrics

import (
	"fmt"

	compbasemetrics "k8s.io/component-base/metrics"
)

const (
	// LLMDRouterEndpointPickerSubsystem is the subsystem for llm-d router endpoint picker metrics.
	LLMDRouterEndpointPickerSubsystem = "llm_d_epp"
)

// HelpMsgWithStability is a helper function to create a help message with stability level.
func HelpMsgWithStability(msg string, stability compbasemetrics.StabilityLevel) string {
	return fmt.Sprintf("[%v] %v", stability, msg)
}

// TokenCountBuckets is a token-count histogram ladder from 1 to ~1M in
// powers of two (with 1 as the low end and 8 as the second entry). Input
// and cached-prompt token histograms across llm-d components share this
// shape.
var TokenCountBuckets = []float64{
	1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
	32768, 65536, 131072, 262144, 524288, 1048576,
}
