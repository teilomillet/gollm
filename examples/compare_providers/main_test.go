package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
)

func TestCreateLLM(t *testing.T) {
	// Test successful creation
	llm, err := createLLM("openai", "gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
	assert.NoError(t, err, "Should create LLM instance")
	assert.NotNil(t, llm, "LLM instance should not be nil")

	// Test with empty API key
	_, err = createLLM("openai", "gpt-4o-mini", "")
	assert.Error(t, err, "Should fail with empty API key")
	assert.Contains(t, err.Error(), "empty API key")

	// Test with invalid model
	_, err = createLLM("openai", "invalid-model", os.Getenv("OPENAI_API_KEY"))
	assert.NoError(t, err, "Should create LLM instance even with invalid model")
}

func TestCompareHelperFunctions(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping compare helper functions test in short mode")
	}

	ctx := context.Background()
	llm1, err := createLLM("openai", "gpt-4o-mini", os.Getenv("OPENAI_API_KEY"))
	assert.NoError(t, err, "Should create first LLM instance")

	llm2, err := createLLM("openai", "gpt-4o", os.Getenv("OPENAI_API_KEY"))
	assert.NoError(t, err, "Should create second LLM instance")

	// Test compareBasicPrompt
	t.Run("compareBasicPrompt", func(t *testing.T) {
		prompt := gollm.NewPrompt("Test basic prompt")
		compareBasicPrompt(ctx, prompt, llm1, llm2)
		// Since the function only logs output, we just verify it doesn't panic
	})

	// Test compareDirectivePrompt
	t.Run("compareDirectivePrompt", func(t *testing.T) {
		prompt := gollm.NewPrompt("Test directive prompt",
			gollm.WithDirectives("Test directive"),
			gollm.WithOutput("Test output:"),
		)
		compareDirectivePrompt(ctx, prompt, llm1, llm2)
		// Since the function only logs output, we just verify it doesn't panic
	})

	// Test compareTemplatePrompt
	t.Run("compareTemplatePrompt", func(t *testing.T) {
		prompt := gollm.NewPrompt("Test template prompt")
		compareTemplatePrompt(ctx, prompt, llm1, llm2)
		// Since the function only logs output, we just verify it doesn't panic
	})
}

func TestCompareProviders(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping compare providers test in short mode")
	}

	// Create LLM clients for different models
	llmGPT3, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	assert.NoError(t, err, "Should create GPT-4o-mini LLM instance")

	llmGPT4, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	assert.NoError(t, err, "Should create GPT-4o LLM instance")

	ctx := context.Background()

	// Test basic prompt comparison
	t.Run("basic_prompt", func(t *testing.T) {
		prompt := gollm.NewPrompt("Explain the concept of machine learning in simple terms.")

		response1, err := llmGPT3.Generate(ctx, prompt)
		assert.NoError(t, err, "Should generate response from GPT-4o-mini")
		assert.NotEmpty(t, response1, "Response from GPT-4o-mini should not be empty")
		assert.Contains(t, strings.ToLower(response1), "learning", "Response should be about machine learning")

		response2, err := llmGPT4.Generate(ctx, prompt)
		assert.NoError(t, err, "Should generate response from GPT-4o")
		assert.NotEmpty(t, response2, "Response from GPT-4o should not be empty")
		assert.Contains(t, strings.ToLower(response2), "learning", "Response should be about machine learning")
	})

	// Test directive prompt comparison
	t.Run("directive_prompt", func(t *testing.T) {
		prompt := gollm.NewPrompt("Explain the concept of blockchain technology",
			gollm.WithDirectives(
				"Use a simple analogy to illustrate",
				"Highlight key features",
				"Mention potential applications",
			),
			gollm.WithOutput("Explanation of blockchain:"),
		)

		response1, err := llmGPT3.Generate(ctx, prompt)
		assert.NoError(t, err, "Should generate response from GPT-4o-mini")
		assert.NotEmpty(t, response1, "Response from GPT-4o-mini should not be empty")
		assert.Contains(t, strings.ToLower(response1), "blockchain", "Response should be about blockchain")
		// Check for output prefix case-insensitively and allow markdown formatting
		assert.True(t, strings.Contains(strings.ToLower(response1), "explanation of blockchain") ||
			strings.Contains(strings.ToLower(response1), "**explanation of blockchain**"),
			"Response should include the output prefix (case-insensitive)")

		response2, err := llmGPT4.Generate(ctx, prompt)
		assert.NoError(t, err, "Should generate response from GPT-4o")
		assert.NotEmpty(t, response2, "Response from GPT-4o should not be empty")
		assert.Contains(t, strings.ToLower(response2), "blockchain", "Response should be about blockchain")
		// Check for output prefix case-insensitively and allow markdown formatting
		assert.True(t, strings.Contains(strings.ToLower(response2), "explanation of blockchain") ||
			strings.Contains(strings.ToLower(response2), "**explanation of blockchain**"),
			"Response should include the output prefix (case-insensitive)")
	})

	// Test template prompt comparison
	t.Run("template_prompt", func(t *testing.T) {
		templatePrompt := gollm.NewPromptTemplate(
			"CustomAnalysis",
			"Analyze a given topic",
			"Analyze the following topic from multiple perspectives: {{.Topic}}",
			gollm.WithPromptOptions(
				gollm.WithDirectives(
					"Consider economic, social, and environmental impacts",
					"Provide pros and cons",
					"Conclude with a balanced summary",
				),
				gollm.WithOutput("Analysis:"),
			),
		)

		// Test JSON schema generation
		schemaBytes, err := llmGPT3.GetPromptJSONSchema()
		assert.NoError(t, err, "Should generate JSON schema")
		assert.NotEmpty(t, schemaBytes, "JSON schema should not be empty")

		var schema map[string]interface{}
		err = json.Unmarshal(schemaBytes, &schema)
		assert.NoError(t, err, "Should parse JSON schema")

		// Check that the schema has either a top-level type or a $ref field
		assert.True(t, schema["type"] != nil || schema["$ref"] != nil,
			"Schema should have either a type or $ref field")

		// Test template execution
		prompt, err := templatePrompt.Execute(map[string]interface{}{
			"Topic": "The adoption of autonomous vehicles",
		})
		assert.NoError(t, err, "Should execute prompt template")
		assert.Contains(t, prompt.Input, "autonomous vehicles", "Prompt should include the topic")

		// Test response generation
		response1, err := llmGPT3.Generate(ctx, prompt)
		assert.NoError(t, err, "Should generate response from GPT-4o-mini")
		assert.NotEmpty(t, response1, "Response from GPT-4o-mini should not be empty")
		assert.Contains(t, strings.ToLower(response1), "autonomous", "Response should be about autonomous vehicles")
		assert.Contains(t, response1, "Analysis:", "Response should include the output prefix")

		response2, err := llmGPT4.Generate(ctx, prompt)
		assert.NoError(t, err, "Should generate response from GPT-4")
		assert.NotEmpty(t, response2, "Response from GPT-4 should not be empty")
		assert.Contains(t, strings.ToLower(response2), "autonomous", "Response should be about autonomous vehicles")
		assert.Contains(t, response2, "Analysis:", "Response should include the output prefix")
	})
}

func TestCompareProvidersErrorHandling(t *testing.T) {
	// Test with invalid API key
	t.Run("invalid_api_key", func(t *testing.T) {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
			gollm.SetAPIKey("invalid_key"),
			gollm.SetMaxTokens(300),
		)
		assert.NoError(t, err, "Should create LLM instance even with invalid key")

		ctx := context.Background()
		prompt := gollm.NewPrompt("Test message")
		_, err = llm.Generate(ctx, prompt)
		assert.Error(t, err, "Should fail with invalid API key")
		assert.Contains(t, err.Error(), "failed to generate after", "Error should indicate generation failure")
	})

	// Test with empty API key
	t.Run("empty_api_key", func(t *testing.T) {
		_, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("gpt-4o-mini"),
			gollm.SetAPIKey(""),
			gollm.SetMaxTokens(300),
		)
		assert.Error(t, err, "Should fail with empty API key")
		assert.Contains(t, err.Error(), "empty API key", "Error should indicate empty API key")
	})

	// Test with invalid model
	t.Run("invalid_model", func(t *testing.T) {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel("invalid-model"),
			gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
			gollm.SetMaxTokens(300),
		)
		assert.NoError(t, err, "Should create LLM instance even with invalid model")

		ctx := context.Background()
		prompt := gollm.NewPrompt("Test message")
		_, err = llm.Generate(ctx, prompt)
		assert.Error(t, err, "Should fail with invalid model")
		assert.Contains(t, err.Error(), "failed to generate after", "Error should indicate generation failure")
	})
}
