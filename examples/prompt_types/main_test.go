package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
)

// cleanJSONResponse removes markdown code block delimiters and trims whitespace
func cleanJSONResponse(response string) string {
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	return strings.TrimSpace(response)
}

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

func TestBasicPromptWithStructuredOutput(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	prompt := gollm.NewPrompt("List the top 3 benefits of exercise",
		gollm.WithOutput("JSON array of benefits, each with a 'title' and 'description'"),
	)

	response, err := llm.Generate(ctx, prompt)
	require.NoError(t, err)
	require.NotEmpty(t, response)

	// Clean the response before parsing
	cleanedResponse := cleanJSONResponse(response)
	var benefits []map[string]string
	err = json.Unmarshal([]byte(cleanedResponse), &benefits)
	require.NoError(t, err)
	assert.Len(t, benefits, 3)

	for _, benefit := range benefits {
		assert.Contains(t, benefit, "title")
		assert.Contains(t, benefit, "description")
	}
}

func TestPromptWithDirectivesAndContext(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	prompt := gollm.NewPrompt("Propose a solution to reduce urban traffic congestion",
		gollm.WithDirectives(
			"Consider both technological and policy-based approaches",
			"Address environmental concerns",
			"Consider cost-effectiveness",
		),
		gollm.WithOutput("Solution proposal in markdown format with headings"),
		gollm.WithContext("The city has a population of 2 million and limited public transportation."),
	)

	response, err := llm.Generate(ctx, prompt)
	require.NoError(t, err)
	require.NotEmpty(t, response)
	assert.Contains(t, response, "#") // Should contain markdown headings
}

func TestPromptTemplate(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	templatePrompt := gollm.NewPromptTemplate(
		"ProductDescription",
		"Generate a product description",
		"Create an engaging product description for a {{.ProductType}} named '{{.ProductName}}'. "+
			"Target audience: {{.TargetAudience}}. Highlight {{.NumFeatures}} key features.",
		gollm.WithPromptOptions(
			gollm.WithDirectives(
				"Use persuasive language",
				"Include a call-to-action",
			),
			gollm.WithOutput("Product description in HTML format"),
		),
	)

	data := map[string]interface{}{
		"ProductType":    "smartwatch",
		"ProductName":    "TimeWise X1",
		"TargetAudience": "fitness enthusiasts",
		"NumFeatures":    3,
	}

	prompt, err := templatePrompt.Execute(data)
	require.NoError(t, err)

	response, err := llm.Generate(ctx, prompt)
	require.NoError(t, err)
	require.NotEmpty(t, response)

	cleanedResponse := cleanJSONResponse(response)
	assert.Contains(t, cleanedResponse, "<") // Should contain HTML tags
}

func TestJSONSchemaValidation(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	prompt := gollm.NewPrompt("Generate a user profile",
		gollm.WithOutput(`{
			"type": "object",
			"properties": {
				"name": {"type": "string"},
				"age": {"type": "integer", "minimum": 18},
				"interests": {"type": "array", "items": {"type": "string"}}
			},
			"required": ["name", "age", "interests"]
		}`),
	)

	response, err := llm.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
	require.NoError(t, err)
	require.NotEmpty(t, response)

	cleanedResponse := cleanJSONResponse(response)
	var result map[string]interface{}
	err = json.Unmarshal([]byte(cleanedResponse), &result)
	require.NoError(t, err)

	// Check the properties field which contains our actual data
	properties, ok := result["properties"].(map[string]interface{})
	require.True(t, ok, "properties field should be a map")

	assert.Contains(t, properties, "name")
	assert.Contains(t, properties, "age")
	assert.Contains(t, properties, "interests")
}

func TestPromptValidation(t *testing.T) {
	validPrompt := gollm.NewPrompt("Valid prompt", gollm.WithMaxLength(1000))
	err := validPrompt.Validate()
	assert.NoError(t, err)

	invalidPrompt := gollm.NewPrompt("", gollm.WithMaxLength(-1))
	err = invalidPrompt.Validate()
	assert.Error(t, err)
}

func TestChainedPrompts(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// First prompt generates a business idea
	ideaPrompt := gollm.NewPrompt("Generate a unique business idea in the sustainability sector",
		gollm.WithOutput("A single sentence business idea"),
	)
	ideaResponse, err := llm.Generate(ctx, ideaPrompt)
	require.NoError(t, err)
	require.NotEmpty(t, ideaResponse)

	// Second prompt analyzes the generated idea
	analysisPrompt := gollm.NewPrompt(
		"Analyze the following business idea: "+cleanJSONResponse(ideaResponse),
		gollm.WithDirectives(
			"Identify potential challenges",
			"Suggest target market",
			"Propose a monetization strategy",
		),
		gollm.WithOutput(`{
			"type": "object",
			"properties": {
				"challenges": {"type": "array", "items": {"type": "string"}},
				"targetMarket": {"type": "string"},
				"monetization": {"type": "string"}
			},
			"required": ["challenges", "targetMarket", "monetization"]
		}`),
	)

	analysisResponse, err := llm.Generate(ctx, analysisPrompt, gollm.WithJSONSchemaValidation())
	require.NoError(t, err)
	require.NotEmpty(t, analysisResponse)

	cleanedResponse := cleanJSONResponse(analysisResponse)
	var result map[string]interface{}
	err = json.Unmarshal([]byte(cleanedResponse), &result)
	require.NoError(t, err)

	// Check the properties field which contains our actual data
	properties, ok := result["properties"].(map[string]interface{})
	require.True(t, ok, "properties field should be a map")

	assert.Contains(t, properties, "challenges")
	assert.Contains(t, properties, "targetMarket")
	assert.Contains(t, properties, "monetization")
}
