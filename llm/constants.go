// Package llm provides core functionality for interacting with Language Learning Models.
package llm

import "time"

// Buffer and streaming constants
const (
	DefaultStreamBufferSize     = 4096 // Default buffer size for streaming responses
	MaxRetryAttempts            = 63   // Maximum retry attempts for exponential backoff
	DefaultOllamaTimeoutSeconds = 5    // Timeout for Ollama endpoint validation
	MinAPIKeyLength             = 20   // Minimum API key length for validation
	MaxValidationSplitParts     = 2    // Maximum parts when splitting validation rules
)

// Timeout durations
const (
	DefaultOllamaTimeout = DefaultOllamaTimeoutSeconds * time.Second
)
