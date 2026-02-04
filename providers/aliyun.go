// Package providers implements LLM provider interfaces and implementations.
package providers

// NewAliyunProvider creates a new Aliyun (Alibaba Cloud DashScope) provider instance.
// Aliyun provides access to Qwen models through an OpenAI-compatible API.
//
// Parameters:
//   - apiKey: Aliyun DashScope API key for authentication
//   - model: The model to use (e.g., "qwen-turbo", "qwen-plus", "qwen-max")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Aliyun Provider instance
//
// Example usage:
//
//	llm, _ := gollm.NewLLM(
//	    gollm.SetProvider("aliyun"),
//	    gollm.SetAPIKey(os.Getenv("DASHSCOPE_API_KEY")),
//	    gollm.SetModel("qwen-turbo"),
//	)
func NewAliyunProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	return NewGenericProvider(apiKey, model, "aliyun", extraHeaders)
}
