package main

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/guiperry/gollm_cerebras"
)

func TestWorkflowConfiguration(t *testing.T) {
	// Test LLM creation with valid configuration
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(500),
	)
	require.NoError(t, err, "Should create LLM with valid configuration")
	require.NotNil(t, llm, "LLM instance should not be nil")

	// Test LLM creation with invalid configuration
	_, err = gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey("invalid-key"),
		gollm.SetMaxTokens(500),
	)
	assert.Error(t, err, "Should fail with invalid API key")
}

func TestResearchPhase(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(500),
	)
	require.NoError(t, err)

	ctx := context.Background()

	// Test research generation with length limit
	researchPrompt := gollm.NewPrompt(
		"Provide a brief overview of quantum computing",
		gollm.WithMaxLength(200),
	)
	research, err := llm.Generate(ctx, researchPrompt)
	require.NoError(t, err, "Research generation should succeed")
	assert.NotEmpty(t, research, "Research should not be empty")

	// Verify research length
	wordCount := len(strings.Fields(research))
	assert.LessOrEqual(t, wordCount, 200, "Research should respect max length")

	// Verify research content
	assert.Contains(t, strings.ToLower(research), "quantum", "Research should contain relevant terms")
}

func TestIdeationPhase(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(500),
	)
	require.NoError(t, err)

	ctx := context.Background()

	// Generate initial research
	researchPrompt := gollm.NewPrompt(
		"Provide a brief overview of quantum computing",
		gollm.WithMaxLength(200),
	)
	research, err := llm.Generate(ctx, researchPrompt)
	require.NoError(t, err)

	// Test ideation with research context
	ideaPrompt := gollm.NewPrompt(
		"Generate 3 article ideas about quantum computing for a general audience",
		gollm.WithContext(research),
	)
	ideas, err := llm.Generate(ctx, ideaPrompt)
	require.NoError(t, err, "Ideation should succeed")
	assert.NotEmpty(t, ideas, "Ideas should not be empty")

	// Verify number of ideas
	ideaLines := strings.Split(ideas, "\n")
	var articleIdeas int
	for _, line := range ideaLines {
		if strings.TrimSpace(line) != "" {
			articleIdeas++
		}
	}
	assert.GreaterOrEqual(t, articleIdeas, 3, "Should generate at least 3 article ideas")
}

func TestRefinementPhase(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(500),
	)
	require.NoError(t, err)

	ctx := context.Background()

	// Generate initial research
	researchPrompt := gollm.NewPrompt(
		"Provide a brief overview of quantum computing",
		gollm.WithMaxLength(200),
	)
	research, err := llm.Generate(ctx, researchPrompt)
	require.NoError(t, err)

	// Test refinement with directives
	refinementPrompt := gollm.NewPrompt(
		"Improve the following paragraph about quantum computing:",
		gollm.WithContext(research),
		gollm.WithDirectives(
			"Use simpler language for a general audience",
			"Add an engaging opening sentence",
			"Conclude with a thought-provoking question",
		),
	)
	refinedParagraph, err := llm.Generate(ctx, refinementPrompt)
	require.NoError(t, err, "Refinement should succeed")
	assert.NotEmpty(t, refinedParagraph, "Refined paragraph should not be empty")

	// Verify directive implementation
	lowerParagraph := strings.ToLower(refinedParagraph)
	assert.True(t, strings.HasSuffix(lowerParagraph, "?"), "Should end with a question")
	assert.NotEqual(t, research, refinedParagraph, "Refined paragraph should be different from research")
}

func TestWorkflowErrorHandling(t *testing.T) {
	// Test with empty API key
	_, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(""),
		gollm.SetMaxTokens(500),
	)
	assert.Error(t, err, "Should fail with empty API key")
}

func TestCompleteWorkflow(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable is not set")
	}

	// Initialize LLM client exactly as in main.go
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(500),
	)
	require.NoError(t, err, "Should create LLM with valid configuration")

	ctx := context.Background()

	// Step 1: Research phase - exactly as in main.go
	researchPrompt := gollm.NewPrompt(
		"Provide a brief overview of quantum computing",
		gollm.WithMaxLength(200),
	)
	research, err := llm.Generate(ctx, researchPrompt)
	require.NoError(t, err, "Research phase should succeed")
	require.NotEmpty(t, research, "Research should not be empty")

	// Verify research constraints and content
	wordCount := len(strings.Fields(research))
	assert.LessOrEqual(t, wordCount, 200, "Research should respect max length")
	lowerResearch := strings.ToLower(research)
	assert.Contains(t, lowerResearch, "quantum", "Research should contain 'quantum'")
	assert.Contains(t, lowerResearch, "comput", "Research should contain 'computer' or 'computing'")

	// Step 2: Ideation phase - use research as context exactly as in main.go
	ideaPrompt := gollm.NewPrompt(
		"Generate 3 article ideas about quantum computing for a general audience",
		gollm.WithContext(research),
	)
	ideas, err := llm.Generate(ctx, ideaPrompt)
	require.NoError(t, err, "Ideation phase should succeed")
	require.NotEmpty(t, ideas, "Ideas should not be empty")

	// Verify ideas content and structure
	lowerIdeas := strings.ToLower(ideas)
	assert.Contains(t, lowerIdeas, "quantum", "Ideas should contain 'quantum'")
	assert.Contains(t, lowerIdeas, "comput", "Ideas should contain 'computer' or 'computing'")

	// Count number of ideas (looking for numbered items or bullet points)
	ideaLines := strings.Split(ideas, "\n")
	var articleIdeas int
	for _, line := range ideaLines {
		line = strings.TrimSpace(line)
		if line != "" && (strings.HasPrefix(line, "-") || strings.HasPrefix(line, "*") ||
			(len(line) > 1 && line[0] >= '1' && line[0] <= '9' && line[1] == '.')) {
			articleIdeas++
		}
	}
	assert.GreaterOrEqual(t, articleIdeas, 3, "Should generate at least 3 article ideas")

	// Step 3: Writing refinement - use research and directives exactly as in main.go
	refinementPrompt := gollm.NewPrompt(
		"Improve the following paragraph about quantum computing:",
		gollm.WithContext(research),
		gollm.WithDirectives(
			"Use simpler language for a general audience",
			"Add an engaging opening sentence",
			"Conclude with a thought-provoking question",
		),
	)
	refinedParagraph, err := llm.Generate(ctx, refinementPrompt)
	require.NoError(t, err, "Refinement phase should succeed")
	require.NotEmpty(t, refinedParagraph, "Refined paragraph should not be empty")

	// Verify refinement content and structure
	lowerParagraph := strings.ToLower(refinedParagraph)
	assert.True(t, strings.HasSuffix(lowerParagraph, "?"), "Should end with a question")
	assert.NotEqual(t, research, refinedParagraph, "Refined paragraph should be different from research")
	assert.Contains(t, lowerParagraph, "quantum", "Refined paragraph should contain 'quantum'")
	assert.Contains(t, lowerParagraph, "comput", "Refined paragraph should contain 'computer' or 'computing'")

	// Verify the content maintains key concepts across all phases
	keyTerms := []string{"quantum", "comput", "bit"}
	for _, term := range keyTerms {
		assert.Contains(t, lowerResearch, term, "Research should contain key term: "+term)
		assert.Contains(t, lowerIdeas, term, "Ideas should contain key term: "+term)
		assert.Contains(t, lowerParagraph, term, "Refined paragraph should contain key term: "+term)
	}
}
