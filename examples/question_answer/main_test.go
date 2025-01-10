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

func TestBasicQuestionAnswer(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	question := "What is photosynthesis?"
	contextInfo := `Photosynthesis is a process used by plants and other organisms to convert light energy 
	into chemical energy that can later be released to fuel the organism's activities. This chemical energy 
	is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.`

	response, err := presets.QuestionAnswer(ctx, llm, question,
		gollm.WithContext(contextInfo),
		gollm.WithMaxLength(150),
	)
	require.NoError(t, err)
	require.NotEmpty(t, response)

	// Check that response contains key concepts from the context
	assert.True(t, strings.Contains(strings.ToLower(response), "light") ||
		strings.Contains(strings.ToLower(response), "energy") ||
		strings.Contains(strings.ToLower(response), "chemical"),
		"Response should mention key concepts from context")
}

func TestQuestionAnswerWithExamples(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	question := "What are the benefits of regular exercise?"
	contextInfo := `Regular physical activity is one of the most important things you can do for your health. 
	Being physically active can improve your brain health, help manage weight, reduce the risk of disease, 
	strengthen bones and muscles, and improve your ability to do everyday activities.`

	response, err := presets.QuestionAnswer(ctx, llm, question,
		gollm.WithContext(contextInfo),
		gollm.WithExamples("Benefit: Improved cardiovascular health, Explanation: Regular exercise strengthens the heart"),
		gollm.WithMaxLength(200),
		gollm.WithDirectives("List benefits with explanations"),
	)
	require.NoError(t, err)
	require.NotEmpty(t, response)

	// Check that response follows the example format
	assert.True(t, strings.Contains(strings.ToLower(response), "benefit") ||
		strings.Contains(response, ":"),
		"Response should follow the example format")
}

func TestQuestionAnswerWithDirectives(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	question := "What are the main features of Python?"
	contextInfo := `Python is a high-level, interpreted programming language known for its simplicity and readability. 
	It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. 
	Python has a large standard library and a vast ecosystem of third-party packages.`

	response, err := presets.QuestionAnswer(ctx, llm, question,
		gollm.WithContext(contextInfo),
		gollm.WithDirectives(
			"List only the top 3 features",
			"Keep the explanation brief",
			"Focus on beginner-friendly aspects",
		),
		gollm.WithMaxLength(150),
	)
	require.NoError(t, err)
	require.NotEmpty(t, response)

	// Check that response mentions at least one key feature from context
	lowercaseResponse := strings.ToLower(response)
	keyFeatures := []string{
		"readability",
		"simple",
		"high-level",
		"interpreted",
		"object-oriented",
		"functional",
		"library",
		"packages",
	}

	foundFeature := false
	for _, feature := range keyFeatures {
		if strings.Contains(lowercaseResponse, feature) {
			foundFeature = true
			break
		}
	}
	assert.True(t, foundFeature, "Response should mention at least one key feature from context")
}

func TestQuestionAnswerErrorHandling(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// Test with invalid provider
	_, err := gollm.NewLLM(
		gollm.SetProvider("nonexistent-provider"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey("test-key"),
	)
	assert.Error(t, err, "Should error with invalid provider")

	// Test with empty context
	response1, err := presets.QuestionAnswer(ctx, llm, "What is the meaning of life?")
	assert.NoError(t, err, "Should work without context")
	assert.NotEmpty(t, response1)

	// Test with multiple examples
	response2, err := presets.QuestionAnswer(ctx, llm, "What is AI?",
		gollm.WithContext("AI is artificial intelligence"),
		gollm.WithExamples(
			"Example 1: This is a test",
			"Example 2: Another test",
			"Example 3: Final test",
		),
	)
	assert.NoError(t, err)
	assert.NotEmpty(t, response2)

	// Test with canceled context
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel() // Cancel immediately
	_, err = presets.QuestionAnswer(cancelCtx, llm, "What is AI?",
		gollm.WithContext("AI is artificial intelligence"),
	)
	assert.Error(t, err, "Should error with canceled context")

	// Test with multiple directives
	response4, err := presets.QuestionAnswer(ctx, llm, "What is AI?",
		gollm.WithContext("AI is artificial intelligence"),
		gollm.WithDirectives(
			"Keep it simple",
			"Use analogies",
			"Be concise",
		),
	)
	assert.NoError(t, err)
	assert.NotEmpty(t, response4)
	assert.NotEqual(t, response2, response4, "Different prompts should yield different responses")
}
