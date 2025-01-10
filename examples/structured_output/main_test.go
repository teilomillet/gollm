package main

import (
	"context"
	"encoding/json"
	"os"
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

// TestPersonInfo for simple tests
type TestPersonInfo struct {
	Name       string   `json:"name" validate:"required"`
	Age        int      `json:"age" validate:"required,gte=0,lte=150"`
	Occupation string   `json:"occupation" validate:"required"`
	Hobbies    []string `json:"hobbies" validate:"required,min=1,max=5"`
}

// ComplexPerson for nested structure tests
type ComplexPerson struct {
	BasicInfo TestPersonInfo `json:"basicInfo" validate:"required"`
	Address   struct {
		Street  string `json:"street" validate:"required"`
		City    string `json:"city" validate:"required"`
		Country string `json:"country" validate:"required"`
	} `json:"address" validate:"required"`
	Contact struct {
		Email     string `json:"email" validate:"required,email"`
		Phone     string `json:"phone" validate:"required"`
		Preferred string `json:"preferred" validate:"required,oneof=email phone"`
	} `json:"contact" validate:"required"`
}

func TestBasicStructuredOutput(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	text := `John Smith is a 32-year-old software engineer from Seattle. 
	He enjoys hiking, photography, and playing guitar in his free time.`

	person, err := presets.ExtractStructuredData[TestPersonInfo](ctx, llm, text,
		gollm.WithDirectives(
			"Extract information into JSON format",
			"Ensure the output is valid JSON",
			"Do not include markdown formatting",
		),
	)
	require.NoError(t, err)
	require.NotNil(t, person)

	// Validate extracted data
	assert.Equal(t, "John Smith", person.Name)
	assert.Equal(t, 32, person.Age)
	assert.Equal(t, "software engineer", person.Occupation)
	assert.GreaterOrEqual(t, len(person.Hobbies), 1)
	assert.LessOrEqual(t, len(person.Hobbies), 5)
}

func TestComplexStructuredOutput(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	text := `Sarah Johnson is a 28-year-old data scientist living at 123 Tech Street, San Francisco, USA. 
	She loves coding, running, and yoga. You can reach her at sarah.j@example.com or +1-555-0123. 
	She prefers to be contacted by email.`

	person, err := presets.ExtractStructuredData[ComplexPerson](ctx, llm, text,
		gollm.WithDirectives(
			"Extract information into JSON format",
			"Ensure the output is valid JSON",
			"Do not include markdown formatting",
			"Include all contact and address information",
		),
	)
	require.NoError(t, err)
	require.NotNil(t, person)

	// Validate basic info
	assert.Equal(t, "Sarah Johnson", person.BasicInfo.Name)
	assert.Equal(t, 28, person.BasicInfo.Age)
	assert.Equal(t, "data scientist", person.BasicInfo.Occupation)
	assert.NotEmpty(t, person.BasicInfo.Hobbies)

	// Validate address
	assert.Equal(t, "123 Tech Street", person.Address.Street)
	assert.Equal(t, "San Francisco", person.Address.City)
	assert.Equal(t, "USA", person.Address.Country)

	// Validate contact
	assert.Equal(t, "sarah.j@example.com", person.Contact.Email)
	assert.Equal(t, "+1-555-0123", person.Contact.Phone)
	assert.Equal(t, "email", person.Contact.Preferred)
}

func TestStructuredOutputValidation(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// Test with insufficient information
	_, err := presets.ExtractStructuredData[TestPersonInfo](ctx, llm, "Just a random text without any relevant information.",
		gollm.WithDirectives("Do not include markdown formatting"),
	)
	assert.Error(t, err, "Should error with insufficient information")

	// Test with invalid age
	text := `Invalid Person is a -5 year old student who likes reading.`
	_, err = presets.ExtractStructuredData[TestPersonInfo](ctx, llm, text,
		gollm.WithDirectives("Do not include markdown formatting"),
	)
	assert.Error(t, err, "Should error with invalid age")

	// Test with too many hobbies
	text = `Bob likes too many things: reading, writing, arithmetic, cooking, baking, gaming, swimming, running, cycling, and yoga.`
	_, err = presets.ExtractStructuredData[TestPersonInfo](ctx, llm, text,
		gollm.WithDirectives("Do not include markdown formatting"),
	)
	assert.Error(t, err, "Should error with too many hobbies")
}

func TestJSONSchemaGeneration(t *testing.T) {
	llm := setupLLM(t)

	// Test schema generation for basic struct
	schema, err := llm.GetPromptJSONSchema(gollm.WithExpandedStruct(true))
	require.NoError(t, err)
	require.NotEmpty(t, schema)

	var schemaMap map[string]interface{}
	err = json.Unmarshal(schema, &schemaMap)
	require.NoError(t, err)

	// Verify schema structure
	assert.Contains(t, schemaMap, "type")
	assert.Contains(t, schemaMap, "properties")
	assert.Contains(t, schemaMap, "required")
}

func TestStructuredOutputErrorHandling(t *testing.T) {
	llm := setupLLM(t)
	ctx := context.Background()

	// Test with empty context
	_, err := presets.ExtractStructuredData[TestPersonInfo](context.TODO(), llm, "some text")
	assert.Error(t, err, "Should error with empty context")

	// Test with nil LLM
	_, err = presets.ExtractStructuredData[TestPersonInfo](ctx, nil, "some text")
	assert.Error(t, err, "Should error with nil LLM")

	// Test with empty text
	_, err = presets.ExtractStructuredData[TestPersonInfo](ctx, llm, "")
	assert.Error(t, err, "Should error with empty text")

	// Test with canceled context
	cancelCtx, cancel := context.WithCancel(ctx)
	cancel()
	_, err = presets.ExtractStructuredData[TestPersonInfo](cancelCtx, llm, "some text")
	assert.Error(t, err, "Should error with canceled context")
}
