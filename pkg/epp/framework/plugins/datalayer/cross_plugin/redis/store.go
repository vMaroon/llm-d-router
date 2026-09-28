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

// Package redis provides a Redis-backed CrossReplicaSyncer.
package redis

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	goredis "github.com/redis/go-redis/v9"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

func init() {
	gob.Register(&attrconcurrency.InFlightLoad{})
}

const (
	RedisStateStoreType = "redis-state-store"
	defaultAddress      = "localhost:6379"
	defaultTTL          = 3 * time.Second
)

var _ fwkdl.CrossReplicaSyncer = (*RedisStateStore)(nil)

type redisConfig struct {
	Address  string `json:"address"`
	Password string `json:"password"`
	DB       int    `json:"db"`
	// TTL bounds how long a replica's published value counts after its last
	// write, so a dead replica's load drops out of the aggregate after TTL.
	TTL string `json:"ttl"`
}

// RedisStateStore is a CrossReplicaSyncer backed by Redis. Each Set writes this
// replica's value into a per-endpoint hash and reads the other replicas' values
// in the same round trip; Get aggregates those cached peer values in memory, so
// the scheduling hot path never waits on Redis. Get returns peers only: the
// caller folds in its own live value.
type RedisStateStore struct {
	typedName fwkplugin.TypedName
	replicaID string
	client    *goredis.Client
	ttl       time.Duration
	cache     sync.Map // hash key -> *peerCacheEntry
}

type peerCacheEntry struct {
	peers     []any
	expiresAt time.Time
}

func RedisStateStoreFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var cfg redisConfig
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("redis-state-store: invalid parameters: %w", err)
		}
	}
	if cfg.Address == "" {
		cfg.Address = defaultAddress
	}
	ttl := defaultTTL
	if cfg.TTL != "" {
		parsed, err := time.ParseDuration(cfg.TTL)
		if err != nil || parsed <= 0 {
			return nil, fmt.Errorf("redis-state-store: invalid ttl %q", cfg.TTL)
		}
		ttl = parsed
	}
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "unknown"
	}
	client := goredis.NewClient(&goredis.Options{Addr: cfg.Address, Password: cfg.Password, DB: cfg.DB})
	ctx := context.Background()
	if handle != nil && handle.Context() != nil {
		ctx = handle.Context()
	}
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis-state-store: failed to connect to Redis at %s: %w", cfg.Address, err)
	}
	return NewRedisStateStore(name, hostname, client, ttl), nil
}

// NewRedisStateStore builds a store for replicaID on an existing client.
func NewRedisStateStore(name, replicaID string, client *goredis.Client, ttl time.Duration) *RedisStateStore {
	return &RedisStateStore{
		typedName: fwkplugin.TypedName{Type: RedisStateStoreType, Name: name},
		replicaID: replicaID,
		client:    client,
		ttl:       ttl,
	}
}

func (s *RedisStateStore) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *RedisStateStore) hashKey(key fwkdl.StateKey, endpointID string) string {
	return string(key) + ":" + endpointID
}

type stampedValue struct {
	Value     any
	WrittenAt time.Time
}

func encodeStamped(value any, now time.Time) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(stampedValue{Value: value, WrittenAt: now}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeStamped(data []byte) (stampedValue, error) {
	var value stampedValue
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&value); err != nil {
		return stampedValue{}, err
	}
	return value, nil
}

// Set publishes this replica's value and caches the fresh values of the other replicas.
func (s *RedisStateStore) Set(ctx context.Context, key fwkdl.StateKey, endpointID string, value any) error {
	now := time.Now()
	data, err := encodeStamped(value, now)
	if err != nil {
		return fmt.Errorf("redis-state-store: encode: %w", err)
	}
	hashKey := s.hashKey(key, endpointID)
	pipe := s.client.TxPipeline()
	pipe.HSet(ctx, hashKey, s.replicaID, data)
	pipe.Expire(ctx, hashKey, s.ttl)
	all := pipe.HGetAll(ctx, hashKey)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("redis-state-store: set and read peers: %w", err)
	}
	raw, err := all.Result()
	if err != nil {
		return fmt.Errorf("redis-state-store: hgetall: %w", err)
	}
	logger := ctrl.LoggerFrom(ctx)
	peers := make([]any, 0, len(raw))
	var stale []string
	for field, encoded := range raw {
		if field == s.replicaID {
			continue
		}
		stamped, err := decodeStamped([]byte(encoded))
		if err != nil {
			logger.V(logutil.DEBUG).Info("redis-state-store: decode error", "field", field, "error", err)
			continue
		}
		if now.Sub(stamped.WrittenAt) > s.ttl {
			stale = append(stale, field)
			continue
		}
		peers = append(peers, stamped.Value)
	}
	if len(stale) > 0 {
		// Dead replicas stop writing; drop their fields so the hash stays small.
		s.client.HDel(ctx, hashKey, stale...)
	}
	s.cache.Store(hashKey, &peerCacheEntry{peers: peers, expiresAt: now.Add(s.ttl)})
	return nil
}

// Get aggregates the peer values cached by the most recent Set. It misses when
// Set has not succeeded within TTL (for example while Redis is unreachable), so
// callers fall back to their local value.
func (s *RedisStateStore) Get(_ context.Context, key fwkdl.StateKey, endpointID string, aggregate func([]any) any) (any, bool, error) {
	hashKey := s.hashKey(key, endpointID)
	cached, ok := s.cache.Load(hashKey)
	if !ok {
		return nil, false, nil
	}
	entry := cached.(*peerCacheEntry)
	if entry.expiresAt.Before(time.Now()) {
		s.cache.CompareAndDelete(hashKey, cached)
		return nil, false, nil
	}
	values := make([]any, len(entry.peers), len(entry.peers)+1)
	copy(values, entry.peers)
	return aggregate(values), true, nil
}

func (s *RedisStateStore) Delete(ctx context.Context, key fwkdl.StateKey, endpointID string) error {
	hashKey := s.hashKey(key, endpointID)
	s.cache.Delete(hashKey)
	return s.client.HDel(ctx, hashKey, s.replicaID).Err()
}
