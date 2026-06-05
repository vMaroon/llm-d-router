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

package tokenizer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

// Only the render backend declares EngineTokenizedPrompt; the estimate backend
// declares the generic key alone. This is what makes a precise prefix-cache
// config without a vllm token-producer fail at startup.
func TestPlugin_ProducesEngineKeyOnlyForRenderBackend(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	estimate, err := NewPlugin(ctx, PluginType, &tokenizerPluginConfig{})
	require.NoError(t, err)
	_, estimateEngine := estimate.Produces()[EngineTokenizedPromptDataKey]
	require.False(t, estimateEngine, "estimate backend must not produce the engine-tokens key")
	_, estimateGeneric := estimate.Produces()[TokenizedPromptDataKey]
	require.True(t, estimateGeneric)

	render, err := NewPlugin(ctx, PluginType, &tokenizerPluginConfig{ModelName: "m"})
	require.NoError(t, err)
	_, renderEngine := render.Produces()[EngineTokenizedPromptDataKey]
	require.True(t, renderEngine, "render backend must produce the engine-tokens key")
	_, renderGeneric := render.Produces()[TokenizedPromptDataKey]
	require.True(t, renderGeneric)
}
