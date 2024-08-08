package gollm

import (
	"github.com/teilomillet/gollm/internal/llm"
)

// Validate checks if the given struct is valid according to its validation rules
func Validate(s interface{}) error {
	return llm.Validate(s)
}

// GenerateJSONSchema generates a JSON schema for the given struct
func GenerateJSONSchema(v interface{}) ([]byte, error) {
	return llm.GenerateJSONSchema(v)
}
