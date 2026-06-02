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
	"encoding/binary"
	"encoding/json"
	"errors"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// bytesPerToken is the fixed bytes-to-pseudo-token ratio. It matches the
// average-characters-per-token assumption used by prefix-cache locality
// scoring, so a block of N pseudo-tokens covers the same bytesPerToken*N input
// bytes a raw-byte block of the same token width would.
const bytesPerToken = 4

// estimateBackend derives pseudo-token IDs directly from request bytes, with no
// real tokenizer. It serializes the user input to a byte stream and packs every
// bytesPerToken bytes into one uint32 token. This is tokenizer-free and covers
// every protocol that can be serialized.
//
// The resulting TokenIDs are meaningful only for content-locality hashing
// (e.g. the approximate prefix-cache scorer's local LRU). They do not
// correspond to any model vocabulary, so they must not be correlated with
// engine-reported KV blocks; that pairing simply never matches and contributes
// nothing rather than mis-routing.
type estimateBackend struct{}

func (estimateBackend) produce(_ context.Context, body *fwkrh.InferenceRequestBody) (*fwkrh.TokenizedPrompt, error) {
	// Pre-tokenized input is already real tokens; pass it through unchanged.
	if body.Generate != nil {
		return &fwkrh.TokenizedPrompt{
			TokenIDs:           body.Generate.TokenIDs,
			MultiModalFeatures: convertMMFeaturesToUpstream(body.Generate.Features),
		}, nil
	}

	raw, err := estimateBytes(body)
	if err != nil {
		return nil, err
	}
	return &fwkrh.TokenizedPrompt{TokenIDs: packBytes(raw)}, nil
}

// estimateBytes serializes the user input of a request body to a byte stream.
// Coverage matches the protocols the approximate prefix-cache scorer handles.
func estimateBytes(body *fwkrh.InferenceRequestBody) ([]byte, error) {
	switch {
	case body.Conversations != nil:
		return json.Marshal(body.Conversations.Items)
	case body.Responses != nil:
		var combined []map[string]any
		if body.Responses.Instructions != nil {
			combined = append(combined, map[string]any{"instructions": body.Responses.Instructions})
		}
		if body.Responses.Tools != nil {
			combined = append(combined, map[string]any{"tools": body.Responses.Tools})
		}
		combined = append(combined, map[string]any{"input": body.Responses.Input})
		return json.Marshal(combined)
	case body.ChatCompletions != nil:
		return chatCompletionsBytes(body.ChatCompletions), nil
	case body.Completions != nil:
		return []byte(body.Completions.Prompt.PlainText()), nil
	case body.Embeddings != nil:
		return json.Marshal(body.Embeddings.Input)
	default:
		return nil, errors.New("unsupported request body type, skipping estimation")
	}
}

// chatCompletionsBytes flattens chat messages (roles + text, with multimodal
// assets folded in on token-aligned boundaries) into a byte stream.
//
// Multimodal assets are keyed by a single content hash here. The per-asset
// placeholder-count weighting currently in the scorer's TokenEstimator would
// move into this function in the full version.
func chatCompletionsBytes(chat *fwkrh.ChatCompletionsRequest) []byte {
	var out []byte
	for _, msg := range chat.Messages {
		if msg.Role != "" {
			out = append(out, []byte(msg.Role)...)
		}
		if msg.Content.Raw != "" {
			out = append(out, []byte(msg.Content.Raw)...)
			continue
		}
		for _, block := range msg.Content.Structured {
			switch block.Type {
			case "text":
				out = append(out, []byte(block.Text)...)
			case "image_url":
				out = align(out)
				h := make([]byte, bytesPerToken)
				binary.LittleEndian.PutUint32(h, uint32(xxhash.Sum64([]byte(block.ImageURL.URL))))
				out = append(out, h...)
			case "video_url":
				out = align(out)
				out = append(out, []byte(block.VideoURL.URL)...)
			case "input_audio", "audio_url":
				out = align(out)
				out = append(out, []byte(block.InputAudio.Data)...)
				out = append(out, []byte(block.InputAudio.Format)...)
			}
		}
	}
	return out
}

// packBytes maps the byte stream to pseudo-token IDs, bytesPerToken bytes each
// (little-endian, zero-padded tail). Reinterpreting the IDs back to bytes
// reproduces the input, so locality hashing over these tokens matches hashing
// over the original bytes.
func packBytes(raw []byte) []uint32 {
	if len(raw) == 0 {
		return nil
	}
	raw = align(raw)
	out := make([]uint32, len(raw)/bytesPerToken)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(raw[i*bytesPerToken:])
	}
	return out
}

// align zero-pads b up to a bytesPerToken boundary.
func align(b []byte) []byte {
	if r := len(b) % bytesPerToken; r != 0 {
		b = append(b, make([]byte, bytesPerToken-r)...)
	}
	return b
}
