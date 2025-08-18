package providers

// Common keys used across multiple providers
const (
	// KeySystemPrompt is the common key for system prompts across providers
	KeySystemPrompt = "system_prompt"

	// KeyTools is the common key for tools/functions across providers
	KeyTools = "tools"

	// KeyStructuredMessages is the common key for structured messages
	KeyStructuredMessages = "structured_messages"

	// KeyToolChoice is the common key for tool choice configuration
	KeyToolChoice = "tool_choice"

	// KeyStructuredResponseSchema is the key to pass a JSON schema for structured responses
	KeyStructuredResponseSchema = "structured_response_schema"
)

// Provider-specific constants
const (
	AnthropicSystemPromptMaxParts = 3    // Maximum parts for splitting system prompts
	OpenRouterCachingThreshold    = 1000 // Minimum message length for prompt caching
	ResponseParserRetryAttempts   = 5    // Retry attempts for response parsing
)
