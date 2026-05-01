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

// Package tokenizer provides DataProducer plugin for the scheduler.
package tokenizer

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	tokenizerTypes "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

type tokenizer interface {
	Render(prompt string) ([]uint32, []tokenizerTypes.Offset, error)
	RenderChat(req *tokenizerTypes.RenderChatRequest) ([]uint32, *tokenization.MultiModalFeatures, error)
}

const (
	// PluginType is the type name used to register the tokenizer plugin.
	PluginType = "tokenizer"

	// TokenizedPromptKey is the data key advertised by this plugin to indicate
	// that it produces tokenized prompt data.
	TokenizedPromptKey = "TokenizedPrompt"

	// TokenizedPromptStateKey is the CycleState key used by the tokenizer scorer
	// to store tokenized prompt data for downstream consumers.
	// Namespaced by PluginType to avoid collisions with other plugins.
	TokenizedPromptStateKey = plugin.StateKey(PluginType + "." + TokenizedPromptKey)
)

// tokenizerPluginConfig holds the configuration for the tokenizer plugin.
//
// Exactly one backend block should be set: udsTokenizerConfig (gRPC over Unix
// domain socket) or vllmHTTPRenderConfig (vLLM HTTP /render). Each backend
// stands on its own. For backward compatibility, an empty configuration falls
// back to udsTokenizerConfig with its default socket path.
type tokenizerPluginConfig struct {
	// TokenizerConfig configures the gRPC-over-UDS backend.
	TokenizerConfig tokenization.UdsTokenizerConfig `json:"udsTokenizerConfig,omitempty"`
	// VLLMHTTPRenderConfig configures the vLLM HTTP /render backend.
	VLLMHTTPRenderConfig *vllmHTTPRenderConfig `json:"vllmHTTPRenderConfig,omitempty"`
	// ModelName is the name of the model whose tokenizer should be loaded.
	ModelName string `json:"modelName"`
}

// PluginFactory is the factory function for the tokenizer plugin.
func PluginFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	config := tokenizerPluginConfig{}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' plugin - %w", PluginType, err)
		}
	}

	if config.ModelName == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: 'modelName' must be specified", PluginType)
	}
	if config.VLLMHTTPRenderConfig != nil && config.TokenizerConfig.IsEnabled() {
		return nil, fmt.Errorf("invalid configuration for '%s' plugin: only one of 'udsTokenizerConfig' or 'vllmHTTPRenderConfig' may be set", PluginType)
	}

	p, err := NewPlugin(handle.Context(), &config)
	if err != nil {
		return nil, err
	}

	return p.WithName(name), nil
}

// NewPlugin creates a new tokenizer plugin instance and constructs the
// configured backend (udsTokenizerConfig or vllmHTTPRenderConfig). When no
// backend block is set, falls back to udsTokenizerConfig with its default
// socket path for backward compatibility.
func NewPlugin(ctx context.Context, config *tokenizerPluginConfig) (*Plugin, error) {
	var (
		tk  tokenizer
		err error
	)
	if config.VLLMHTTPRenderConfig != nil {
		tk, err = newVLLMHTTPRenderer(config.VLLMHTTPRenderConfig, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize vLLM HTTP renderer for '%s' plugin - %w", PluginType, err)
		}
	} else {
		tk, err = tokenization.NewUdsTokenizer(ctx, &config.TokenizerConfig, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize UDS tokenizer for '%s' plugin - %w", PluginType, err)
		}
	}

	return &Plugin{
		typedName: plugin.TypedName{Type: PluginType},
		tokenizer: tk,
	}, nil
}

// TokenizedPromptState holds the tokenization result for a single request,
// stored in CycleState for consumption by downstream scorers.
// This follows the standard IGW pattern where scorers share data via CycleState
// (same as NoHitLRU reading from prefix-cache scorer).
//
// Single-prompt requests (Prompt.Raw / chat completions) populate TokenIDs and
// optionally MMFeatures. Multi-prompt requests (OpenAI-style Prompt.Strings
// arrays) populate TokenIDsList — one token sequence per prompt, no MM
// features. Consumers should prefer Sequences()/TotalTokens() over reading
// the fields directly so they handle both shapes uniformly.
type TokenizedPromptState struct {
	TokenIDs     []uint32
	TokenIDsList [][]uint32
	MMFeatures   *tokenization.MultiModalFeatures
}

// Sequences returns the prompt token sequences regardless of which field is
// populated. Returns nil when no tokens are present.
func (t *TokenizedPromptState) Sequences() [][]uint32 {
	if t == nil {
		return nil
	}
	if len(t.TokenIDsList) > 0 {
		return t.TokenIDsList
	}
	if len(t.TokenIDs) > 0 {
		return [][]uint32{t.TokenIDs}
	}
	return nil
}

// TotalTokens returns the total token count across all sequences.
func (t *TokenizedPromptState) TotalTokens() int {
	if t == nil {
		return 0
	}
	if len(t.TokenIDsList) > 0 {
		n := 0
		for _, seq := range t.TokenIDsList {
			n += len(seq)
		}
		return n
	}
	return len(t.TokenIDs)
}

// Clone implements plugin.StateData.
func (t *TokenizedPromptState) Clone() plugin.StateData {
	if t == nil {
		return nil
	}
	var ids []uint32
	if t.TokenIDs != nil {
		ids = make([]uint32, len(t.TokenIDs))
		copy(ids, t.TokenIDs)
	}
	var list [][]uint32
	if t.TokenIDsList != nil {
		list = make([][]uint32, len(t.TokenIDsList))
		for i, seq := range t.TokenIDsList {
			cp := make([]uint32, len(seq))
			copy(cp, seq)
			list[i] = cp
		}
	}
	return &TokenizedPromptState{
		TokenIDs:     ids,
		TokenIDsList: list,
		MMFeatures:   cloneMMFeatures(t.MMFeatures),
	}
}

// CompletionPrompts returns the list of prompts to process from a completions
// request. Single-prompt requests yield one element; OpenAI-style multi-prompt
// requests (Prompt.Strings) yield N elements. Empty entries are dropped. When
// both Raw and Strings are populated, Raw takes precedence.
func CompletionPrompts(p scheduling.Prompt) []string {
	if p.Raw != "" {
		return []string{p.Raw}
	}
	if len(p.Strings) == 0 {
		return nil
	}
	out := make([]string, 0, len(p.Strings))
	for _, s := range p.Strings {
		if s == "" {
			continue
		}
		out = append(out, s)
	}
	return out
}

// cloneMMFeatures deep-copies the maps/slices so cloned CycleState entries
// are fully independent and safe from concurrent mutation.
func cloneMMFeatures(src *tokenization.MultiModalFeatures) *tokenization.MultiModalFeatures {
	if src == nil {
		return nil
	}
	dst := &tokenization.MultiModalFeatures{}
	if src.MMHashes != nil {
		dst.MMHashes = make(map[string][]string, len(src.MMHashes))
		for k, v := range src.MMHashes {
			cp := make([]string, len(v))
			copy(cp, v)
			dst.MMHashes[k] = cp
		}
	}
	if src.MMPlaceholders != nil {
		dst.MMPlaceholders = make(map[string][]kvblock.PlaceholderRange, len(src.MMPlaceholders))
		for k, v := range src.MMPlaceholders {
			cp := make([]kvblock.PlaceholderRange, len(v))
			copy(cp, v)
			dst.MMPlaceholders[k] = cp
		}
	}
	return dst
}

// Plugin tokenizes the prompt in the incoming request and stores
// the result in CycleState for downstream consumers (scorers).
type Plugin struct {
	typedName plugin.TypedName
	tokenizer tokenizer
}

// TypedName returns the typed name of the plugin.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// WithName sets the name of the plugin.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// tokenize renders the request prompts and returns a TokenizedPromptState.
// Returns nil on error, when the request has no body, or when the request type
// is unsupported. Multi-prompt completions (Prompt.Strings) are rendered one
// sequence per prompt and surfaced via TokenIDsList.
func (p *Plugin) tokenize(ctx context.Context, request *scheduling.LLMRequest) *TokenizedPromptState {
	logger := log.FromContext(ctx).WithName(p.typedName.String())
	traceLogger := logger.V(logging.TRACE)

	if request.Body == nil {
		traceLogger.Info("Request body is nil, skipping tokenization")
		return nil
	}

	traceLogger.Info("Request body present",
		"hasCompletions", request.Body.Completions != nil,
		"hasChatCompletions", request.Body.ChatCompletions != nil)

	switch {
	case request.Body.Completions != nil:
		prompts := CompletionPrompts(request.Body.Completions.Prompt)
		if len(prompts) == 0 {
			traceLogger.Info("Completions request has no usable prompt, skipping tokenization")
			return nil
		}
		if len(prompts) == 1 {
			traceLogger.Info("Calling Render for completions", "promptLength", len(prompts[0]))
			tokenIDs, _, err := p.tokenizer.Render(prompts[0])
			if err != nil {
				logger.Error(err, "Tokenization failed, skipping")
				return nil
			}
			traceLogger.Info("Tokenization succeeded", "tokenCount", len(tokenIDs))
			return &TokenizedPromptState{TokenIDs: tokenIDs}
		}
		traceLogger.Info("Calling Render per prompt for multi-prompt completions", "promptCount", len(prompts))
		list := make([][]uint32, 0, len(prompts))
		for i, pr := range prompts {
			ids, _, err := p.tokenizer.Render(pr)
			if err != nil {
				logger.Error(err, "Tokenization failed for prompt, skipping request", "promptIndex", i)
				return nil
			}
			list = append(list, ids)
		}
		traceLogger.Info("Multi-prompt tokenization succeeded",
			"promptCount", len(list), "totalTokens", totalTokenCount(list))
		return &TokenizedPromptState{TokenIDsList: list}

	case request.Body.ChatCompletions != nil:
		renderReq := ChatCompletionsToRenderChatRequest(request.Body.ChatCompletions)
		traceLogger.Info("Calling RenderChat for chat completions", "messageCount", len(request.Body.ChatCompletions.Messages))
		tokenIDs, mmFeatures, err := p.tokenizer.RenderChat(renderReq)
		if err != nil {
			logger.Error(err, "Tokenization failed, skipping")
			return nil
		}
		traceLogger.Info("Tokenization succeeded", "tokenCount", len(tokenIDs))
		return &TokenizedPromptState{TokenIDs: tokenIDs, MMFeatures: mmFeatures}

	default:
		traceLogger.Info("Unsupported request type, skipping tokenization")
		return nil
	}
}

func totalTokenCount(seqs [][]uint32) int {
	n := 0
	for _, s := range seqs {
		n += len(s)
	}
	return n
}

// ChatCompletionsToRenderChatRequest converts a ChatCompletionsRequest to a
// tokenization RenderChatRequest, including multimodal content blocks.
func ChatCompletionsToRenderChatRequest(chat *scheduling.ChatCompletionsRequest) *tokenizerTypes.RenderChatRequest {
	conversation := make([]tokenizerTypes.Conversation, 0, len(chat.Messages))
	for _, msg := range chat.Messages {
		conv := tokenizerTypes.Conversation{
			Role:    msg.Role,
			Content: tokenizerTypes.Content{Raw: msg.Content.Raw},
		}
		for _, block := range msg.Content.Structured {
			conv.Content.Structured = append(conv.Content.Structured, tokenizerTypes.ContentBlock{
				Type:     block.Type,
				Text:     block.Text,
				ImageURL: tokenizerTypes.ImageBlock{URL: block.ImageURL.Url},
			})
		}
		conversation = append(conversation, conv)
	}

	return &tokenizerTypes.RenderChatRequest{
		Conversation:              conversation,
		Tools:                     chat.Tools,
		Documents:                 chat.Documents,
		ChatTemplate:              chat.ChatTemplate,
		ReturnAssistantTokensMask: chat.ReturnAssistantTokensMask,
		ContinueFinalMessage:      chat.ContinueFinalMessage,
		AddGenerationPrompt:       chat.AddGenerationPrompt,
		ChatTemplateKWArgs:        chat.ChatTemplateKWArgs,
	}
}
