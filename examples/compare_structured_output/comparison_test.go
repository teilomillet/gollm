package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/guiperry/gollm_cerebras"
	"github.com/guiperry/gollm_cerebras/presets"
)

// validateComplexPerson validates the ComplexPerson struct according to the defined rules
func validateComplexPerson(person ComplexPerson) error {
	if person.Age < 0 || person.Age > 150 {
		return fmt.Errorf("age must be between 0 and 150")
	}
	if len(person.Hobbies) < 1 || len(person.Hobbies) > 5 {
		return fmt.Errorf("number of hobbies must be between 1 and 5")
	}
	if person.LuckyNumber < 1 || person.LuckyNumber > 100 {
		return fmt.Errorf("lucky number must be between 1 and 100")
	}
	return nil
}

func TestComplexPersonValidation(t *testing.T) {
	tests := []struct {
		name    string
		person  ComplexPerson
		wantErr bool
	}{
		{
			name: "valid person",
			person: ComplexPerson{
				Name:          "John Doe",
				Age:           30,
				Occupation:    "Software Engineer",
				City:          "San Francisco",
				Country:       "USA",
				FavoriteColor: "Blue",
				Hobbies:       []string{"Reading", "Hiking"},
				Education:     "Bachelor's Degree",
				PetName:       "Max",
				LuckyNumber:   42,
			},
			wantErr: false,
		},
		{
			name: "invalid age - too high",
			person: ComplexPerson{
				Name:          "John Doe",
				Age:           200,
				Occupation:    "Software Engineer",
				City:          "San Francisco",
				Country:       "USA",
				FavoriteColor: "Blue",
				Hobbies:       []string{"Reading", "Hiking"},
				Education:     "Bachelor's Degree",
				PetName:       "Max",
				LuckyNumber:   42,
			},
			wantErr: true,
		},
		{
			name: "invalid hobbies - too many",
			person: ComplexPerson{
				Name:          "John Doe",
				Age:           30,
				Occupation:    "Software Engineer",
				City:          "San Francisco",
				Country:       "USA",
				FavoriteColor: "Blue",
				Hobbies:       []string{"Reading", "Hiking", "Gaming", "Cooking", "Swimming", "Dancing"},
				Education:     "Bachelor's Degree",
				PetName:       "Max",
				LuckyNumber:   42,
			},
			wantErr: true,
		},
		{
			name: "invalid lucky number - too high",
			person: ComplexPerson{
				Name:          "John Doe",
				Age:           30,
				Occupation:    "Software Engineer",
				City:          "San Francisco",
				Country:       "USA",
				FavoriteColor: "Blue",
				Hobbies:       []string{"Reading", "Hiking"},
				Education:     "Bachelor's Degree",
				PetName:       "Max",
				LuckyNumber:   150,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateComplexPerson(tt.person)
			if tt.wantErr {
				assert.Error(t, err, "validateComplexPerson() should return error")
			} else {
				assert.NoError(t, err, "validateComplexPerson() should not return error")
			}
		})
	}
}

func TestJSONSchemaGeneration(t *testing.T) {
	schema, err := gollm.GenerateJSONSchema(ComplexPerson{})
	assert.NoError(t, err, "Should generate JSON schema without error")
	assert.NotEmpty(t, schema, "Generated schema should not be empty")

	// Parse the schema to verify its structure
	var schemaMap map[string]interface{}
	err = json.Unmarshal(schema, &schemaMap)
	assert.NoError(t, err, "Should parse schema as JSON")

	// Check required fields
	required, ok := schemaMap["required"].([]interface{})
	assert.True(t, ok, "Schema should have required fields")
	assert.Contains(t, required, "name", "Name should be required")
	assert.Contains(t, required, "age", "Age should be required")
	assert.Contains(t, required, "occupation", "Occupation should be required")
}

func TestCompareModels(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping model comparison test in short mode")
	}

	ctx := context.Background()

	// Create test configurations
	models := []struct {
		provider string
		model    string
	}{
		{"openai", "gpt-4o-mini"},
		{"anthropic", "claude-3-5-haiku-latest"},
	}

	configs := make([]*gollm.Config, 0, len(models))
	for _, m := range models {
		apiKeyEnv := fmt.Sprintf("%s_API_KEY", strings.ToUpper(m.provider))
		apiKey := os.Getenv(apiKeyEnv)
		if apiKey == "" {
			t.Logf("Skipping %s %s: API key not set", m.provider, m.model)
			continue
		}

		config := gollm.NewConfig()
		gollm.SetProvider(m.provider)(config)
		gollm.SetModel(m.model)(config)
		gollm.SetAPIKey(apiKey)(config)
		gollm.SetMaxTokens(500)(config)
		gollm.SetMaxRetries(3)(config)
		gollm.SetRetryDelay(time.Second * 2)(config)
		gollm.SetLogLevel(gollm.LogLevelDebug)(config)

		configs = append(configs, config)
	}

	if len(configs) == 0 {
		t.Skip("No valid configurations available. Please set API keys.")
	}

	// Generate JSON schema
	schema, err := gollm.GenerateJSONSchema(ComplexPerson{})
	assert.NoError(t, err)

	// Create prompt
	promptText := fmt.Sprintf(`Generate information about a fictional person.
Create a fictional person with the following attributes: name, age, occupation, city, country, favorite color, hobbies (1-5), education, pet name, and lucky number (1-100).
Ensure all fields are filled and adhere to the specified constraints.
Return the data as a JSON object that adheres to this schema:
%s`, string(schema))

	// Compare models
	results, err := presets.CompareModels(ctx, promptText, validateComplexPerson, configs...)
	assert.NoError(t, err)
	assert.NotEmpty(t, results)

	// Test analysis
	analysis := presets.AnalyzeComparisonResults(results)
	assert.NotEmpty(t, analysis)
}

func TestCompareModelsErrorHandling(t *testing.T) {
	ctx := context.Background()

	// Test with invalid configuration
	t.Run("invalid_config", func(t *testing.T) {
		config := gollm.NewConfig()
		gollm.SetProvider("openai")(config)
		gollm.SetModel("gpt-4o-mini")(config)
		gollm.SetAPIKey("invalid_key")(config)
		gollm.SetMaxTokens(500)(config)
		gollm.SetMaxRetries(3)(config)
		gollm.SetRetryDelay(time.Second * 2)(config)
		gollm.SetLogLevel(gollm.LogLevelDebug)(config)

		_, err := presets.CompareModels[ComplexPerson](ctx, "test prompt", validateComplexPerson, config)
		assert.Error(t, err, "Should fail with invalid configuration")
		assert.Contains(t, err.Error(), "failed to generate")
	})

	// Test with empty prompt
	t.Run("empty_prompt", func(t *testing.T) {
		config := gollm.NewConfig()
		gollm.SetProvider("openai")(config)
		gollm.SetModel("gpt-4o-mini")(config)
		if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			t.Skip("Skipping test: OPENAI_API_KEY not set")
		} else {
			gollm.SetAPIKey(apiKey)(config)
		}

		_, err := presets.CompareModels[ComplexPerson](ctx, "", validateComplexPerson, config)
		assert.Error(t, err, "Should fail with empty prompt")
		assert.Contains(t, err.Error(), "prompt cannot be empty")
	})

	// Test with nil validator
	t.Run("nil_validator", func(t *testing.T) {
		config := gollm.NewConfig()
		gollm.SetProvider("openai")(config)
		gollm.SetModel("gpt-4o-mini")(config)
		if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			t.Skip("Skipping test: OPENAI_API_KEY not set")
		} else {
			gollm.SetAPIKey(apiKey)(config)
		}

		_, err := presets.CompareModels[ComplexPerson](ctx, "test prompt", nil, config)
		assert.Error(t, err, "Should fail with nil validator")
		assert.Contains(t, err.Error(), "validator function cannot be nil")
	})
}
