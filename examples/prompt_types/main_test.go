package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
)

// cleanResponse removes markdown code block delimiters and trims whitespace
func cleanResponse(response string) string {
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimPrefix(response, "```")
	response = strings.TrimSuffix(response, "```")
	return strings.TrimSpace(response)
}

func setupLLM(t *testing.T) gollm.LLM {
	t.Helper()
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
	cleanedResponse := cleanResponse(response.AsText())
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

	data := map[string]any{
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

	cleanedResponse := cleanResponse(response.AsText())
	assert.Contains(t, cleanedResponse, "<") // Should contain HTML tags
}

func TestJSONSchemaValidation(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	prompt := gollm.NewPrompt("Generate a user profile")

	response, err := llm.Generate(ctx, prompt, gollm.WithStructuredResponseSchema[*UserProfile]())
	require.NoError(t, err)
	require.NotEmpty(t, response)

	var result map[string]any
	err = json.Unmarshal([]byte(response.AsText()), &result)
	require.NoError(t, err)

	// Check the properties field which contains our actual data
	properties, ok := result["properties"].(map[string]any)
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
		"Analyze the following business idea: "+cleanResponse(ideaResponse.AsText()),
		gollm.WithDirectives(
			"Identify potential challenges",
			"Suggest target market",
			"Propose a monetization strategy",
		),
	)

	analysisResponse, err := llm.Generate(ctx, analysisPrompt, gollm.WithStructuredResponseSchema[*IdeaAnalysis]())
	require.NoError(t, err)
	require.NotEmpty(t, analysisResponse)
	require.NoError(t, err)
	require.NotEmpty(t, analysisResponse)

	// Print response for debugging
	fmt.Printf("Raw analysis response: %s\n", analysisResponse.AsText())

	cleanedResponse := cleanResponse(analysisResponse.AsText())

	fmt.Printf("Cleaned analysis response: %s\n", cleanedResponse)

	var result map[string]any
	err = json.Unmarshal([]byte(cleanedResponse), &result)
	require.NoError(t, err)

	// Print the unmarshalled result
	resultBytes, _ := json.MarshalIndent(result, "", "  ")
	fmt.Printf("Unmarshalled result:\n%s\n", string(resultBytes))

	// The response is a JSON schema with the actual fields inside the "properties" object
	properties, ok := result["properties"].(map[string]any)
	require.True(t, ok, "Response should contain a properties field of type map")

	// Check the fields in the properties object
	assert.Contains(t, properties, "challenges", "Properties should contain challenges field")
	assert.Contains(t, properties, "targetMarket", "Properties should contain targetMarket field")
	assert.Contains(t, properties, "monetization", "Properties should contain monetization field")
}

type IdeaAnalysis struct {
	Challenges   []string `json:"challenges"`
	TargetMarket string   `json:"targetMarket"`
	Monetization string   `json:"monetization"`
}
