package preciseprefixcache

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

const udsSocketPath = "/tmp/tokenizer/tokenizer-uds.socket"

// skipIfNoUDSTokenizer skips the test if UDS tokenizer socket is not available.
func skipIfNoUDSTokenizer(t *testing.T) {
	if _, err := os.Stat(udsSocketPath); os.IsNotExist(err) {
		t.Skipf("UDS tokenizer socket not available at %s, skipping test", udsSocketPath)
	}
}

// createUDSTokenizer creates a UDS tokenizer for testing.
func createUDSTokenizer(t *testing.T, model string) *tokenization.UdsTokenizer {
	udsTokenizer, err := tokenization.NewUdsTokenizer(context.Background(),
		&tokenization.UdsTokenizerConfig{SocketFile: udsSocketPath}, model)
	require.NoError(t, err)
	return udsTokenizer
}

const mmModelName = "Qwen/Qwen2-VL-2B-Instruct"

// TestRenderChat_MultimodalContent_UDS exercises the full MM pipeline through
// the real UDS tokenizer with a multimodal model:
//   - RenderChat with structured content (text + image_url)
//   - Verify MultiModalFeatures are returned (MMHashes, MMPlaceholders)
//   - Compute BlockExtraFeatures from MM features
//   - Compute block keys with extraFeatures taint
//   - Verify tainted block keys differ from text-only block keys
func TestRenderChat_MultimodalContent_UDS(t *testing.T) {
	skipIfNoUDSTokenizer(t)

	udsTokenizer := createUDSTokenizer(t, mmModelName)
	defer func() {
		err := udsTokenizer.Close()
		require.NoError(t, err)
	}()

	mmRenderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{
				Role: "user",
				Content: types.Content{
					Structured: []types.ContentBlock{
						{Type: "image_url", ImageURL: types.ImageBlock{URL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg=="}},
						{Type: "text", Text: "What do you see in this image? Please describe it in detail."},
					},
				},
			},
		},
		AddGenerationPrompt: true,
	}

	tokens, mmFeatures, err := udsTokenizer.RenderChat(mmRenderReq)
	require.NoError(t, err, "RenderChat with MM content should not error")
	require.NotEmpty(t, tokens, "Should produce tokens from MM content")
	require.NotNil(t, mmFeatures, "MultiModalFeatures should be non-nil for multimodal input")
	require.NotEmpty(t, mmFeatures.MMHashes, "MMHashes should contain at least one modality")
	require.NotEmpty(t, mmFeatures.MMPlaceholders, "MMPlaceholders should contain at least one modality")

	t.Logf("MM RenderChat: %d tokens, modalities=%v", len(tokens), func() []string {
		keys := make([]string, 0, len(mmFeatures.MMHashes))
		for k := range mmFeatures.MMHashes {
			keys = append(keys, k)
		}
		return keys
	}())

	// Compute BlockExtraFeatures from MM features.
	blockSize := kvblock.DefaultTokenProcessorConfig().BlockSize
	extraFeatures := kvblock.ComputeBlockExtraFeatures(
		mmFeatures.MMHashes, mmFeatures.MMPlaceholders,
		blockSize, len(tokens))
	require.NotNil(t, extraFeatures, "ComputeBlockExtraFeatures should produce non-nil result for MM input")

	// Verify at least one block has MM taint.
	hasTaint := false
	for _, ef := range extraFeatures {
		if ef != nil && len(ef.MMHashes) > 0 {
			hasTaint = true
			break
		}
	}
	require.True(t, hasTaint, "At least one block should have MM hash taint")

	// Compute block keys WITH extra features (MM-tainted).
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	mmBlockKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, mmModelName, extraFeatures)
	require.NoError(t, err)
	require.NotEmpty(t, mmBlockKeys)

	// Compute block keys WITHOUT extra features (text-only view of same tokens).
	textBlockKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, mmModelName, nil)
	require.NoError(t, err)
	require.Equal(t, len(mmBlockKeys), len(textBlockKeys), "Same token count should produce same number of blocks")

	// At least one tainted block should produce a different hash than text-only.
	differ := false
	for i := range mmBlockKeys {
		if mmBlockKeys[i] != textBlockKeys[i] {
			differ = true
			t.Logf("Block %d hashes differ: mm=%x text=%x", i, mmBlockKeys[i], textBlockKeys[i])
			break
		}
	}
	require.True(t, differ, "MM-tainted block keys must differ from text-only block keys")
}

// TestRenderChat_TextOnly_NoMMFeatures_UDS verifies that RenderChat with plain
// text content returns nil MMFeatures (no false positives).
func TestRenderChat_TextOnly_NoMMFeatures_UDS(t *testing.T) {
	skipIfNoUDSTokenizer(t)

	udsTokenizer := createUDSTokenizer(t, mmModelName)
	defer func() {
		err := udsTokenizer.Close()
		require.NoError(t, err)
	}()

	textRenderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{
				Role:    "user",
				Content: types.Content{Raw: "Hello, how are you doing today?"},
			},
		},
		AddGenerationPrompt: true,
	}

	tokens, mmFeatures, err := udsTokenizer.RenderChat(textRenderReq)
	require.NoError(t, err)
	require.NotEmpty(t, tokens)
	require.Nil(t, mmFeatures, "Text-only RenderChat should return nil MultiModalFeatures")
}

// TestMMPipeline_ScoreTokensWithExtraFeatures_UDS is an end-to-end test that
// exercises the full MM-aware scoring pipeline through the real indexer:
//   - Tokenize MM content via UDS
//   - Compute block keys with MM taint
//   - Populate the index with tainted keys
//   - Score via ScoreTokens with extraFeatures
//   - Verify pods with tainted entries score higher
func TestMMPipeline_ScoreTokensWithExtraFeatures_UDS(t *testing.T) {
	skipIfNoUDSTokenizer(t)

	ctx := utils.NewTestContext(t)

	// 1. Tokenize MM content.
	udsTokenizer := createUDSTokenizer(t, mmModelName)
	defer func() {
		err := udsTokenizer.Close()
		require.NoError(t, err)
	}()

	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{
				Role: "user",
				Content: types.Content{
					Structured: []types.ContentBlock{
						{Type: "image_url", ImageURL: types.ImageBlock{URL: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg=="}},
						{Type: "text", Text: "Describe the contents of this photograph."},
					},
				},
			},
		},
		AddGenerationPrompt: true,
	}

	tokens, mmFeatures, err := udsTokenizer.RenderChat(renderReq)
	require.NoError(t, err)
	require.NotEmpty(t, tokens)
	require.NotNil(t, mmFeatures)

	// 2. Compute extra features and block keys.
	tpConfig := kvblock.DefaultTokenProcessorConfig()
	extraFeatures := kvblock.ComputeBlockExtraFeatures(
		mmFeatures.MMHashes, mmFeatures.MMPlaceholders,
		tpConfig.BlockSize, len(tokens))
	require.NotNil(t, extraFeatures)

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(tpConfig)
	require.NoError(t, err)

	mmBlockKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, mmModelName, extraFeatures)
	require.NoError(t, err)
	require.GreaterOrEqual(t, len(mmBlockKeys), 2, "Need at least 2 blocks for scoring test")

	// 3. Set up indexer.
	kvcacheConfig, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	kvcacheConfig.TokenizersPoolConfig = &tokenization.Config{
		ModelName:    mmModelName,
		WorkersCount: 1,
		UdsTokenizerConfig: &tokenization.UdsTokenizerConfig{
			SocketFile: udsSocketPath,
		},
	}

	prefixCacheScorer, err := New(ctx, PluginConfig{
		IndexerConfig:  kvcacheConfig,
		KVEventsConfig: kvevents.DefaultConfig(),
	})
	require.NoError(t, err)

	// 4. Populate index with MM-tainted block keys.
	//    pod-a has all blocks, pod-b has only the first.
	kvBlockIndex := prefixCacheScorer.kvCacheIndexer.KVBlockIndex()
	for i, key := range mmBlockKeys {
		pods := []kvblock.PodEntry{{PodIdentifier: "10.0.0.1:8080"}}
		if i == 0 {
			pods = append(pods, kvblock.PodEntry{PodIdentifier: "10.0.0.2:8080"})
		}
		err := kvBlockIndex.Add(ctx, []kvblock.BlockHash{kvblock.EmptyBlockHash}, []kvblock.BlockHash{key}, pods)
		require.NoError(t, err)
	}

	// 5. Score with extraFeatures — pod-a should score higher.
	endpoints := []scheduling.Endpoint{
		scheduling.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-a"},
				Address:        "10.0.0.1",
				Port:           "8080",
			},
			nil, nil,
		),
		scheduling.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-b"},
				Address:        "10.0.0.2",
				Port:           "8080",
			},
			nil, nil,
		),
	}

	// Write tokenized state with MM features to CycleState (simulating tokenizer plugin).
	cycleState := scheduling.NewCycleState()
	cycleState.Write(tokenizer.TokenizedPromptStateKey, &tokenizer.TokenizedPromptState{
		TokenIDs:   tokens,
		MMFeatures: mmFeatures,
	})

	request := &scheduling.LLMRequest{
		RequestId:   "test-mm-e2e",
		TargetModel: mmModelName,
	}

	scores := prefixCacheScorer.Score(ctx, cycleState, request, endpoints)

	gotByAddress := make(map[string]float64)
	for endpoint, score := range scores {
		if m := endpoint.GetMetadata(); m != nil {
			gotByAddress[fmt.Sprintf("%s:%s", m.Address, m.Port)] = score
		}
	}

	t.Logf("MM E2E scores: %v", gotByAddress)
	require.Contains(t, gotByAddress, "10.0.0.1:8080")
	require.Contains(t, gotByAddress, "10.0.0.2:8080")
	require.Greater(t, gotByAddress["10.0.0.1:8080"], gotByAddress["10.0.0.2:8080"],
		"pod-a (all MM-tainted blocks) should score higher than pod-b (only first block)")
}
