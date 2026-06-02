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
	"unsafe"

	"github.com/cespare/xxhash/v2"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
)

// hashTokens hashes a token block the way the scorer's HashBlock does: uint32s
// reinterpreted as little-endian bytes.
func hashTokens(t []uint32) uint64 {
	if len(t) == 0 {
		return 0
	}
	return xxhash.Sum64(unsafe.Slice((*byte)(unsafe.Pointer(&t[0])), len(t)*4))
}

// TestPackBytes_KeyPreserving asserts packed-token hashing matches raw-byte
// hashing, so the scorer's cache keys are unchanged.
func TestPackBytes_KeyPreserving(t *testing.T) {
	raw := []byte("the quick brown fox jumps over!!") // len 32, 4-byte aligned
	if len(raw)%bytesPerToken != 0 {
		t.Fatalf("fixture must be %d-byte aligned, got len %d", bytesPerToken, len(raw))
	}
	tokens := packBytes(raw)
	if got, want := len(tokens), len(raw)/bytesPerToken; got != want {
		t.Fatalf("token count: got %d, want %d", got, want)
	}
	if hashTokens(tokens) != xxhash.Sum64(raw) {
		t.Errorf("packed-token hash != raw-byte hash; estimate path is not key-preserving")
	}
}

// TestEstimateBackend_GeneratePassthrough asserts pre-tokenized input is kept
// as real tokens, not re-estimated.
func TestEstimateBackend_GeneratePassthrough(t *testing.T) {
	in := []uint32{7, 8, 9}
	tp, err := estimateBackend{}.produce(context.Background(), &fwkrh.InferenceRequestBody{
		Generate: &fwkrh.GenerateRequest{TokenIDs: in},
	})
	if err != nil {
		t.Fatalf("produce: %v", err)
	}
	if len(tp.TokenIDs) != len(in) {
		t.Fatalf("got %d tokens, want %d", len(tp.TokenIDs), len(in))
	}
	for i := range in {
		if tp.TokenIDs[i] != in[i] {
			t.Errorf("token %d: got %d, want %d", i, tp.TokenIDs[i], in[i])
		}
	}
}

// TestEstimateBackend_CompletionsDeterministic asserts the same prompt produces
// the same tokens (locality precondition) and that distinct prompts differ.
func TestEstimateBackend_CompletionsDeterministic(t *testing.T) {
	body := func(s string) *fwkrh.InferenceRequestBody {
		return &fwkrh.InferenceRequestBody{Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: s}}}
	}
	a, err := estimateBackend{}.produce(context.Background(), body("hello world"))
	if err != nil {
		t.Fatalf("produce a: %v", err)
	}
	b, err := estimateBackend{}.produce(context.Background(), body("hello world"))
	if err != nil {
		t.Fatalf("produce b: %v", err)
	}
	if hashTokens(a.TokenIDs) != hashTokens(b.TokenIDs) {
		t.Error("same prompt produced different tokens")
	}
	c, err := estimateBackend{}.produce(context.Background(), body("hello there"))
	if err != nil {
		t.Fatalf("produce c: %v", err)
	}
	if hashTokens(a.TokenIDs) == hashTokens(c.TokenIDs) {
		t.Error("distinct prompts produced identical tokens")
	}
}
