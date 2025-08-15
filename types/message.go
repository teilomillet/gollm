// Package types contains shared type definitions used across the gollm library.
// It helps avoid import cycles while providing common data structures.
package types

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
