package llm

// GenerateOption is a function type for configuring generation behavior.
type GenerateOption func(*GenerateConfig)

// WithStructuredResponseSchema configures Generate to produce output conforming to the provided schema type.
// The generic type parameter T should be a struct type describing the expected JSON structure.
func WithStructuredResponseSchema[T any]() GenerateOption {
	return func(cfg *GenerateConfig) {
		cfg.StructuredResponseSchema = *new(T)
	}
}

// WithStructuredResponse configures Generate to produce output conforming to the provided schema value.
// Use this when you already have a JSON Schema or example instance at runtime (e.g., map[string]any or a struct instance).
func WithStructuredResponse(schema any) GenerateOption {
	return func(cfg *GenerateConfig) {
		cfg.StructuredResponseSchema = schema
	}
}

// WithStreamBufferSize sets the size of the token buffer for streaming responses.
func WithStreamBufferSize(size int) GenerateOption {
	return func(cfg *GenerateConfig) {
		cfg.StreamBufferSize = size
	}
}

// WithRetryStrategy defines how to handle stream interruptions.
func WithRetryStrategy(strategy RetryStrategy) GenerateOption {
	return func(cfg *GenerateConfig) {
		cfg.RetryStrategy = strategy
	}
}

// GenerateConfig holds configuration options for text generation.
type GenerateConfig struct {
	// StreamBufferSize is the size of the token buffer
	StreamBufferSize int

	// RetryStrategy defines how to handle stream interruptions
	RetryStrategy RetryStrategy

	// StructuredResponseSchema, when non-nil, requests that the response conform to the provided schema.
	// Providers that support JSON Schema will receive it directly; others will have the schema
	// embedded into the prompt, and the result validated client-side.
	StructuredResponseSchema any
}
