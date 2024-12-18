package gollm

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestLiveAPICallWithGenerate(t *testing.T) {

	t.Run("OpenAI Live Test", func(t *testing.T) {
		const testOpenAIDefaultModel = "gpt-3.5-turbo"
		// Using non-standard environment variable to prevent accidental use
		apiKey := os.Getenv("TEST_OPENAI_API_KEY")
		if apiKey == "" {
			t.Skip("TEST_OPENAI_API_KEY is not set")
		}

		ctx := context.Background()
		llm, err := NewLLM(
			SetProvider("openai"),
			SetModel(testOpenAIDefaultModel),
			SetAPIKey(apiKey),
			SetMaxTokens(200),
			SetMaxRetries(3),
			SetRetryDelay(time.Second*2),
			SetLogLevel(LogLevelInfo),
		)
		assert.NoError(t, err)

		response, err := llm.Generate(ctx, NewPrompt("Say hello in 3 languages"))
		assert.NoError(t, err)
		assert.NotEmpty(t, response)
	})

	t.Run("Ollama Live Test", func(t *testing.T) {
		ollamaURL := os.Getenv("OLLAMA_API_BASE")
		if ollamaURL == "" {
			ollamaURL = "http://localhost:11434" // default Ollama endpoint
		}

		// Check if Ollama is accessible and get available models
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		client := &http.Client{}
		req, err := http.NewRequestWithContext(ctx, "GET", ollamaURL+"/api/tags", nil)
		if err != nil {
			t.Skip("Failed to create request to Ollama endpoint:", err)
		}

		resp, err := client.Do(req)
		if err != nil {
			t.Skip("Ollama endpoint is not accessible:", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			t.Skip("Ollama endpoint returned non-200 status:", resp.StatusCode)
		}

		// Parse the response to get available models
		var tagsResponse struct {
			Models []struct {
				Name string `json:"name"`
			} `json:"models"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&tagsResponse); err != nil {
			t.Skip("Failed to decode models response:", err)
		}

		if len(tagsResponse.Models) == 0 {
			t.Skip("No models available at Ollama endpoint")
		}

		// Find the first non-embedding model
		var chatModel string
		for _, model := range tagsResponse.Models {
			if !strings.Contains(strings.ToLower(model.Name), "embed") {
				chatModel = model.Name
				break
			}
		}

		if chatModel == "" {
			t.Skip("No suitable models available at Ollama endpoint")
		}

		llm, err := NewLLM(
			SetProvider("ollama"),
			SetModel(chatModel), // Use the first available chat model
			SetMaxTokens(200),
			SetMaxRetries(3),
			SetRetryDelay(time.Second*2),
			SetLogLevel(LogLevelInfo),
		)
		assert.NoError(t, err)

		ctx = context.Background()
		response, err := llm.Generate(ctx, NewPrompt("Say hello in 3 languages"))
		assert.NoError(t, err)
		assert.NotEmpty(t, response)
	})
}
