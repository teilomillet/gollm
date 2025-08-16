// Package models provides shared model definitions for the GoLLM library.
// These models are used across multiple packages to avoid import cycles.
package models

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
type MemoryMessage struct {
	Metadata     map[string]any
	Role         string
	Content      string
	CacheControl string
	Tokens       int
}
