package anthropic

import (
	"context"
	"testing"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/stretchr/testify/require"
)

func TestSplitUsageTotal(t *testing.T) {
	parser := NewAnthropicParser()
	var usage fwkrh.Usage
	for _, frame := range []string{
		"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20}}}\n\n",
		"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":10}}\n\n",
	} {
		response, err := parser.ParseResponse(context.Background(), []byte(frame), map[string]string{"content-type": "text/event-stream"}, false)
		require.NoError(t, err)
		require.NotNil(t, response.Usage)
		usage.MergeCumulative(*response.Usage)
	}
	require.Equal(t, 20, usage.PromptTokens)
	require.Equal(t, 10, usage.CompletionTokens)
	require.Equal(t, 30, usage.TotalTokens)
	usage.MergeCumulative(fwkrh.Usage{CompletionTokens: 5, TotalTokens: 5})
	require.Equal(t, 30, usage.TotalTokens, "duplicates and partial records cannot decrease the total")
}
