// Package gollm provides streaming functionality for Language Learning Models.
// This file contains type definitions and re-exports for working with streaming responses.
package gollm

import (
	"github.com/weave-labs/gollm/llm"
)

// Re-export streaming types from the llm package
type (
	// TokenStream represents a stream of tokens from an LLM response.
	// It follows Go's io.ReadCloser pattern but at the token level.
	TokenStream = llm.TokenStream

	// StreamToken represents a single token from a streaming response.
	StreamToken = llm.StreamToken

	// RetryStrategy defines the interface for handling stream interruptions.
	RetryStrategy = llm.RetryStrategy
)
