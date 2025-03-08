package gollm

import (
	"testing"
)

func TestOllamaWithoutAPIKey(t *testing.T) {
	// Test creating an LLM with Ollama provider without an API key
	_, err := NewLLM(
		SetProvider("ollama"),
		SetModel("llama3.2"),
	)

	if err != nil {
		t.Fatalf("Failed to create LLM with Ollama provider without API key: %v", err)
	}
}
