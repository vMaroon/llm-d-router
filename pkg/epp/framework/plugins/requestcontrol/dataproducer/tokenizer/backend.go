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
	"errors"
	"fmt"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// tokenInputProducer turns a parsed request body into the tokenized,
// KV-cache-affecting representation (TokenizedPrompt). Implementations differ
// only in fidelity: renderBackend calls a real tokenizer; estimateBackend packs
// bytes. The plugin and downstream consumers store the result and never branch
// on which backend produced it.
//
// The input is the per-protocol request structs of InferenceRequestBody. Once
// the unified prompt representation (UnifiedPrompt/TokenInputs, #1380) is
// populated by the parsers, a backend's input collapses to that single shape;
// the contract here is unchanged.
type tokenInputProducer interface {
	produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error)
}

// renderBackend produces token IDs from a real tokenizer. It owns the
// protocol-to-tokenizer dispatch, including the pre-tokenized (Generate)
// passthrough, so neither the plugin nor downstream consumers see it.
type renderBackend struct {
	tk tokenizer
}

func (b renderBackend) produce(ctx context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error) {
	switch {
	case body.Completions != nil:
		tokenIDs, _, err := b.tk.Render(ctx, body.Completions.Prompt.Raw)
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		return &fwkrh.TokenizedPrompt{TokenIDs: tokenIDs}, nil
	case body.ChatCompletions != nil:
		tokenIDs, mmFeatures, err := b.tk.RenderChat(ctx, ChatCompletionsToRenderChatRequest(body.ChatCompletions))
		if err != nil {
			return nil, fmt.Errorf("tokenization failed: %w", err)
		}
		return &fwkrh.TokenizedPrompt{TokenIDs: tokenIDs, MultiModalFeatures: convertMMFeaturesToUpstream(mmFeatures)}, nil
	case body.Generate != nil:
		return &fwkrh.TokenizedPrompt{
			TokenIDs:           body.Generate.TokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}, nil
	default:
		return nil, errors.New("unsupported request body type, skipping tokenization")
	}
}
