// Package types contains shared type definitions used across the gollm library.
// It helps avoid import cycles while providing common data structures.
package types

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
type MemoryMessage struct {
	Role         string                 // Role of the message sender (e.g., "user", "assistant")
	Content      string                 // The actual message content
	Tokens       int                    // Number of tokens in the message
	CacheControl string                 // Caching strategy for this message ("ephemeral", "persistent", etc.)
	Metadata     map[string]interface{} // Additional provider-specific metadata
}
