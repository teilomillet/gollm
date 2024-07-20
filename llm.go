// File: goal/llm.go

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
func NewLLM(configPath string, logLevel ...string) (LLM, error) {
	// Initialize logging first
	if len(logLevel) > 0 {
		llm.InitLogging(logLevel[0])
	} else {
		llm.InitLogging("warn") // Default to warn level
	}

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
