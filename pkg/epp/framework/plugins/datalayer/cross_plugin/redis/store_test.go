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

package redis

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

const key = fwkdl.StateKey("inflight:test")

func sumLoads(values []any) any {
	total := &attrconcurrency.InFlightLoad{}
	for _, v := range values {
		l := v.(*attrconcurrency.InFlightLoad)
		total.Requests += l.Requests
		total.Tokens += l.Tokens
	}
	return total
}

func newPair(t *testing.T, ttl time.Duration) (*miniredis.Miniredis, *RedisStateStore, *RedisStateStore) {
	t.Helper()
	mr := miniredis.RunT(t)
	client := func() *goredis.Client {
		c := goredis.NewClient(&goredis.Options{Addr: mr.Addr(), MaxRetries: -1})
		t.Cleanup(func() { _ = c.Close() })
		return c
	}
	return mr, NewRedisStateStore("redis", "replica-a", client(), ttl), NewRedisStateStore("redis", "replica-b", client(), ttl)
}

func load(requests, tokens int64) *attrconcurrency.InFlightLoad {
	return &attrconcurrency.InFlightLoad{Requests: requests, Tokens: tokens}
}

func TestGetReturnsPeersOnly(t *testing.T) {
	ctx := context.Background()
	_, a, b := newPair(t, time.Minute)

	require.NoError(t, a.Set(ctx, key, "ns/ep", load(3, 300)))
	require.NoError(t, b.Set(ctx, key, "ns/ep", load(5, 500)))
	require.NoError(t, a.Set(ctx, key, "ns/ep", load(4, 400)))

	got, ok, err := a.Get(ctx, key, "ns/ep", sumLoads)
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, load(5, 500), got, "a sees b's value, not its own")

	got, ok, err = b.Get(ctx, key, "ns/ep", sumLoads)
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, load(3, 300), got, "b last read a's first value")

	_, ok, err = a.Get(ctx, key, "ns/other", sumLoads)
	require.NoError(t, err)
	assert.False(t, ok, "no Set for this endpoint yet")
}

func TestStalePeerDropsOut(t *testing.T) {
	ctx := context.Background()
	mr, a, b := newPair(t, 50*time.Millisecond)

	require.NoError(t, b.Set(ctx, key, "ns/ep", load(9, 900)))
	time.Sleep(80 * time.Millisecond)
	require.NoError(t, a.Set(ctx, key, "ns/ep", load(1, 100)))

	got, ok, err := a.Get(ctx, key, "ns/ep", sumLoads)
	require.NoError(t, err)
	require.True(t, ok)
	assert.Equal(t, load(0, 0), got, "b stopped writing longer than ttl ago")
	fields, err := mr.HKeys("inflight:test:ns/ep")
	require.NoError(t, err)
	assert.Equal(t, []string{"replica-a"}, fields, "stale field removed")
}

func TestGetMissesWhenRedisIsDown(t *testing.T) {
	ctx := context.Background()
	mr, a, b := newPair(t, 100*time.Millisecond)

	require.NoError(t, b.Set(ctx, key, "ns/ep", load(2, 200)))
	require.NoError(t, a.Set(ctx, key, "ns/ep", load(1, 100)))
	_, ok, _ := a.Get(ctx, key, "ns/ep", sumLoads)
	require.True(t, ok)

	mr.Close()
	assert.Error(t, a.Set(ctx, key, "ns/ep", load(1, 100)))
	time.Sleep(150 * time.Millisecond)
	_, ok, err := a.Get(ctx, key, "ns/ep", sumLoads)
	require.NoError(t, err)
	assert.False(t, ok, "caller falls back to its local value after ttl")
}

func TestDeleteRemovesOwnField(t *testing.T) {
	ctx := context.Background()
	mr, a, b := newPair(t, time.Minute)

	require.NoError(t, a.Set(ctx, key, "ns/ep", load(1, 100)))
	require.NoError(t, b.Set(ctx, key, "ns/ep", load(2, 200)))
	require.NoError(t, a.Delete(ctx, key, "ns/ep"))

	fields, err := mr.HKeys("inflight:test:ns/ep")
	require.NoError(t, err)
	assert.Equal(t, []string{"replica-b"}, fields)
	_, ok, _ := a.Get(ctx, key, "ns/ep", sumLoads)
	assert.False(t, ok)
}
