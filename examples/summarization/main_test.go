package main

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/presets"
)

func setupLLM(t *testing.T) gollm.LLM {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	require.NoError(t, err)
	return llm
}

func TestBasicSummarization(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	text := `Artificial intelligence (AI) is transforming various sectors of society,
	including healthcare, finance, and transportation. While AI offers numerous
	benefits such as improved efficiency and decision-making, it also raises
	concerns about privacy, job displacement, and ethical considerations.`

	summary, err := presets.Summarize(ctx, llm, text,
		gollm.WithMaxLength(50),
		gollm.WithDirectives(
			"Focus on main impacts",
			"Include both benefits and concerns",
		),
	)
	require.NoError(t, err)
	require.NotEmpty(t, summary)

	// Verify summary contains key concepts
	lowercaseSummary := strings.ToLower(summary)
	keyTerms := []string{
		"ai", "artificial intelligence",
		"healthcare", "finance", "transportation",
		"benefits", "concerns", "privacy", "ethical",
	}

	foundTerms := 0
	for _, term := range keyTerms {
		if strings.Contains(lowercaseSummary, term) {
			foundTerms++
		}
	}
	assert.GreaterOrEqual(t, foundTerms, 3, "Summary should mention at least 3 key concepts")
}

func TestTechnicalSummarization(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	text := `The study examined the effects of climate change on biodiversity in tropical rainforests.
	Research conducted over a 10-year period (2010-2020) showed a 15% decline in species diversity,
	with amphibians being the most affected group showing a 30% population decrease. Temperature
	increases of 2°C were correlated with changes in breeding patterns and migration timing.
	The findings suggest that immediate conservation measures are necessary to prevent further
	biodiversity loss.`

	summary, err := presets.Summarize(ctx, llm, text,
		gollm.WithMaxLength(200),
		gollm.WithDirectives(
			"Maintain technical accuracy",
			"Include key statistics",
			"Preserve time periods",
			"Highlight main findings",
		),
		gollm.WithOutput("Technical Summary:"),
	)
	require.NoError(t, err)
	require.NotEmpty(t, summary)

	// Verify technical details are preserved
	lowercaseSummary := strings.ToLower(summary)
	assert.Contains(t, lowercaseSummary, "15%", "Should preserve species decline percentage")
	assert.Contains(t, lowercaseSummary, "30%", "Should preserve amphibian decline percentage")
	assert.Contains(t, lowercaseSummary, "2°c", "Should preserve temperature data")
	assert.True(t, strings.Contains(lowercaseSummary, "2010") ||
		strings.Contains(lowercaseSummary, "10-year"),
		"Should mention study period")
}

func TestSummarizationWithCustomOutput(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	text := `The company's Q4 financial report shows revenue growth of 25%,
	exceeding market expectations. Operating costs decreased by 10% due to
	automation initiatives. Customer satisfaction scores improved from
	85% to 92%, while employee retention rate remained stable at 95%.`

	summary, err := presets.Summarize(ctx, llm, text,
		gollm.WithDirectives(
			"Structure as bullet points",
			"Focus on key metrics",
			"Compare with previous values",
		),
		gollm.WithOutput("Q4 Performance Summary:"),
	)
	require.NoError(t, err)
	require.NotEmpty(t, summary)

	// Verify output format and content
	assert.Contains(t, summary, "Q4 Performance Summary:", "Should include custom output header")
	assert.Contains(t, summary, "25%", "Should include revenue growth")
	assert.Contains(t, summary, "92%", "Should include satisfaction score")
}

func TestSummarizationErrorHandling(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// Test with invalid LLM configuration
	_, err := gollm.NewLLM(
		gollm.SetProvider("nonexistent-provider"),
		gollm.SetModel("invalid-model"),
		gollm.SetAPIKey("test-key"),
	)
	require.Error(t, err, "Should error with invalid provider")

	// Test with nil context (should use background context)
	summary, err := presets.Summarize(nil, llm, "Test text")
	require.NoError(t, err, "Should not error with nil context")
	require.NotEmpty(t, summary, "Should generate summary with nil context")

	// Test with canceled context
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel()
	_, err = presets.Summarize(
		cancelCtx,
		llm,
		"This is a test text that should not be summarized because the context is canceled.",
	)
	require.Error(t, err, "Should error with canceled context")

	// Test with nil LLM
	_, err = presets.Summarize(ctx, nil, "Test text")
	require.Error(t, err, "Should error with nil LLM")
}
