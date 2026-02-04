// Package providers implements LLM provider interfaces and implementations.
package providers

// NewLMStudioProvider creates a new LM Studio provider instance.
// LM Studio provides a local LLM server with an OpenAI-compatible API.
//
// Parameters:
//   - apiKey: Not required for LM Studio, can be any string
//   - model: The model identifier (e.g., "lfm2.5-1.2b-instruct-mlx")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured LM Studio Provider instance
//
// Example usage:
//
//	llm, _ := gollm.NewLLM(
//	    gollm.SetProvider("lmstudio"),
//	    gollm.SetModel("lfm2.5-1.2b-instruct-mlx"),
//	)
func NewLMStudioProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "lmstudio", extraHeaders)
}
