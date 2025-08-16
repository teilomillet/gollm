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
// Use this when you already have a JSON Schema or example instance at runtime (e.g., map[string]any or a struct
// instance).
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
	RetryStrategy            RetryStrategy
	StructuredResponseSchema any
	StreamBufferSize         int
}
