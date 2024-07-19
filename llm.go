package goal

import (
	"context"

	"github.com/teilomillet/goal/internal/llm"
)

// LLM represents the main interface for interacting with language models
type LLM interface {
	Generate(ctx context.Context, prompt string) (response string, fullPrompt string, err error)
	SetOption(key string, value interface{})
}

type llmImpl struct {
	llm.LLM
}

// NewLLM creates a new LLM instance from a config file
func NewLLM(configPath string) (LLM, error) {
	config, err := llm.LoadConfig(configPath)
	if err != nil {
		return nil, err
	}
	l, err := llm.NewLLMFromConfig(config)
	if err != nil {
		return nil, err
	}
	return &llmImpl{l}, nil
}
