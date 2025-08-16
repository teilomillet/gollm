// Package gollm provides validation functionality for Language Learning Model interactions.
// This file contains utilities for validating structured data and generating JSON schemas,
// which are essential for ensuring proper data formats in LLM communications.
package gollm

import (
	"fmt"

	"github.com/weave-labs/gollm/llm"
)

// Validate checks if the given struct is valid according to its validation rules.
// It uses struct tags to define validation rules and performs comprehensive validation
// of the input structure.
//
// The function supports various validation rules through struct tags, including:
//   - required: Field must be present and non-zero
//   - min/max: Numeric range validation
//   - len: Exact length requirement
//   - email: Email format validation
//   - url: URL format validation
//   - regex: Pattern matching
//   - contains/excludes: AsText content validation
//   - unique: Array unique items validation
//   - minItems/maxItems: Array length validation
//   - password: Password strength validation
//
// Example usage:
//
//	type Config struct {
//	    Model     string `validate:"required,model"`
//	    MaxTokens int    `validate:"min=1,max=4096"`
//	    Email     string `validate:"required,email"`
//	    Password  string `validate:"required,password=strong"`
//	}
//
//	config := Config{
//	    Model:     "gpt-4",
//	    MaxTokens: 2048,
//	    Email:     "user@example.com",
//	    Password:  "SecureP@ss123",
//	}
//	err := Validate(&config)
//
// Parameters:
//   - s: The struct to validate. Must be a pointer to a struct.
//
// Returns:
//   - error: nil if validation passes, otherwise returns detailed validation errors
func Validate(s any) error {
	if err := llm.Validate(s); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}
	return nil
}

// GenerateJSONSchema generates a JSON schema for the given struct.
// The schema is generated based on struct fields and their tags, providing
// a complete JSON Schema that can be used for validation or documentation.
//
// The function analyzes struct fields and generates a schema that includes:
//   - Field types and formats
//   - Required fields
//   - Validation rules from struct tags
//   - Nested object structures
//   - Array specifications
//   - Custom validation rules
//   - Format constraints
//
// Example usage:
//
//	type Message struct {
//	    Role     string   `json:"role" validate:"required,oneof=system user assistant"`
//	    Content  string   `json:"content" validate:"required,min=1"`
//	    Tokens   int      `json:"tokens,omitempty" validate:"min=0"`
//	    Tags     []string `json:"tags,omitempty" validate:"unique"`
//	}
//
//	type Conversation struct {
//	    ID       string    `json:"id" validate:"required,uuid"`
//	    Messages []Message `json:"messages" validate:"required,min=1"`
//	    Model    string    `json:"model" validate:"required,model"`
//	}
//
//	schema, err := GenerateJSONSchema(&Conversation{})
//
// Parameters:
//   - v: The struct to generate schema for. Must be a pointer to a struct.
//
// Returns:
//   - []byte: The generated JSON schema as a byte slice
//   - error: Any error encountered during schema generation
func GenerateJSONSchema(v any) ([]byte, error) {
	schema, err := llm.GenerateJSONSchema(v)
	if err != nil {
		return nil, fmt.Errorf("failed to generate JSON schema: %w", err)
	}
	return schema, nil
}
