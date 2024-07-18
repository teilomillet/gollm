package main

import (
	"context"
	"fmt"
	"os"

	"github.com/teilomillet/goal/llm"
	"go.uber.org/zap"
)

func main() {
	llm.SetLogLevel(zap.InfoLevel)

	providers := []struct {
		name  string
		model string
	}{
		{"anthropic", "claude-3-opus-20240229"},
		{"openai", "gpt-4"},
	}

	prompt := "What are the advantages of using Go for backend development?"

	for _, p := range providers {
		apiKey := os.Getenv(fmt.Sprintf("%s_API_KEY", p.name))
		llmProvider, err := llm.GetProvider(p.name, apiKey, p.model)
		if err != nil {
			llm.Logger.Error("Error creating LLM provider", zap.String("provider", p.name), zap.Error(err))
			continue
		}

		llmClient := llm.NewLLM(llmProvider)
		llmClient.SetOption("temperature", 0.7)
		llmClient.SetOption("max_tokens", 150)

		ctx := context.Background()
		response, err := llmClient.Generate(ctx, prompt)
		if err != nil {
			llm.Logger.Error("Error generating text", zap.String("provider", p.name), zap.Error(err))
			continue
		}

		fmt.Printf("%s response:\n%s\n\n", p.name, response)
	}
}
