// File: structured_data_extractor.go

package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/teilomillet/gollm/llm"
)

// ExtractStructuredData extracts structured data from text based on a given struct type
func ExtractStructuredData[T any](ctx context.Context, l llm.LLM, text string, opts ...llm.PromptOption) (*T, error) {
	structType := reflect.TypeOf((*T)(nil)).Elem()
	schema, err := llm.GenerateJSONSchema(reflect.New(structType).Interface())
	if err != nil {
		return nil, fmt.Errorf("failed to generate JSON schema: %w", err)
	}

	prompt := llm.NewPrompt(
		fmt.Sprintf("Extract the following information from the given text:\n\n%s\n\nRespond with a JSON object matching this schema:\n%s", text, string(schema)),
		append(opts,
			llm.WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output matches the provided JSON schema exactly",
				"If a field cannot be confidently filled, leave it as null or an empty string/array as appropriate",
			),
			llm.WithOutput("JSON object matching the provided schema"),
		)...,
	)

	// Convert the prompt to a string before passing it to Generate
	promptString := prompt.String()

	response, _, err := l.Generate(ctx, promptString)
	if err != nil {
		return nil, fmt.Errorf("failed to generate structured data: %w", err)
	}

	var result T
	if err := json.Unmarshal([]byte(response), &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if err := llm.Validate(result); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	return &result, nil
}
