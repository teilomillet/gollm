package llm

import (
	"fmt"
	"os"
)

// NewLLM creates a new instance of an LLM based on the provider name
func NewLLM(provider, model string) (LLM, error) {
	var (
		apiKey string
		llm    LLM
	)

	switch provider {
	case "openai":
		apiKey = os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("OPENAI_API_KEY not set in environment")
		}
		llm = NewOpenAILLM(apiKey, model)

	case "anthropic":
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("ANTHROPIC_API_KEY not set in environment")
		}
		llm = NewAnthropicLLM(apiKey, model)

	case "groq":
		apiKey = os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("GROQ_API_KEY not set in environment")
		}
		llm = NewGroqLLM(apiKey, model)

	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}

	return llm, nil
}
