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

package datalayer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

type countCloneable int

func (c countCloneable) Clone() fwkdl.Cloneable { return c }

// peerSyncer returns fixed peer values, or misses when peers is nil.
type peerSyncer struct{ peers []any }

func (s peerSyncer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "peer-syncer", Name: "peer-syncer"}
}
func (s peerSyncer) Set(context.Context, fwkdl.StateKey, string, any) error { return nil }
func (s peerSyncer) Get(_ context.Context, _ fwkdl.StateKey, _ string, aggregate func([]any) any) (any, bool, error) {
	if s.peers == nil {
		return nil, false, nil
	}
	return aggregate(append([]any(nil), s.peers...)), true, nil
}
func (s peerSyncer) Delete(context.Context, fwkdl.StateKey, string) error { return nil }

func TestCrossReplicaAttribute(t *testing.T) {
	local := 3
	spec := fwkdl.CrossReplicaSpec{
		StateKey: "inflight:test",
		Supply: func(string) func() fwkdl.Cloneable {
			return func() fwkdl.Cloneable { return countCloneable(local) }
		},
		Aggregate: func(values []any) any {
			sum := 0
			for _, v := range values {
				sum += int(v.(countCloneable))
			}
			return countCloneable(sum)
		},
	}

	t.Run("peers plus live local value", func(t *testing.T) {
		attr := crossReplicaAttribute(context.Background(), peerSyncer{peers: []any{countCloneable(4), countCloneable(5)}}, spec, "ns/ep")
		assert.Equal(t, countCloneable(12), attr.Get())
		local = 7
		assert.Equal(t, countCloneable(16), attr.Get(), "local value is read live on every Get")
		local = 3
	})

	t.Run("no peers yet counts local only", func(t *testing.T) {
		attr := crossReplicaAttribute(context.Background(), peerSyncer{peers: []any{}}, spec, "ns/ep")
		assert.Equal(t, countCloneable(3), attr.Get())
	})

	t.Run("syncer miss falls back to local", func(t *testing.T) {
		attr := crossReplicaAttribute(context.Background(), peerSyncer{}, spec, "ns/ep")
		assert.Equal(t, countCloneable(3), attr.Get())
	})
}
