package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/assess"
)

func TestStreaming(t *testing.T) {
	test := assess.NewTest(t).
		WithProviders(map[string]string{
			"anthropic": "claude-3-5-haiku-latest",
			"openai":    "gpt-4o-mini",
		}).
		WithBatchConfig(assess.BatchTestConfig{
			EnableBatch:  true,
			MaxParallel:  1,
			BatchTimeout: 5 * time.Minute,
		})

	// Test basic streaming functionality
	t.Run("basic_streaming", func(t *testing.T) {
		test.AddCase("basic_stream", "Write something.").
			WithTimeout(30 * time.Second).
			Validate(func(response string) error {
				if response == "" {
					return assert.AnError
				}
				return nil
			})
	})

	// Test streaming with function calls
	t.Run("function_calls", func(t *testing.T) {
		tools := []gollm.Tool{
			{
				Function: gollm.Function{
					Name:        "get_weather",
					Description: "Get the current weather",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		}

		test.AddCase("stream_with_tools", "What's the weather like in San Francisco?").
			WithTools(tools).
			WithTimeout(30 * time.Second).
			Validate(func(response string) error {
				if response == "" {
					return assert.AnError
				}
				return nil
			})
	})

	// Test streaming cancellation
	t.Run("cancellation", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		defer cancel()

		test.AddCase("cancelled_stream", "Write an extremely long story that will take several minutes to complete.").
			WithTimeout(1 * time.Second).
			Validate(func(response string) error {
				select {
				case <-ctx.Done():
					return nil // Expected behavior
				case <-time.After(2 * time.Second):
					return assert.AnError
				}
			})
	})

	// Run all tests in batch mode
	ctx := context.Background()
	test.RunBatch(ctx)
}

// Helper function to collect stream tokens
func collectStreamTokens(stream gollm.TokenStream, ctx context.Context) ([]string, error) {
	var tokens []string
	for {
		token, err := stream.Next(ctx)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		tokens = append(tokens, token.Text)
	}
	return tokens, nil
}

// TestProviderStreaming tests the actual streaming implementation of each provider
func TestProviderStreaming(t *testing.T) {
	providers := map[string]string{
		"anthropic": "claude-3-5-haiku-latest",
		"openai":    "gpt-4o-mini",
	}

	for providerName, model := range providers {
		t.Run(providerName, func(t *testing.T) {
			// Get API key from environment
			apiKeyEnv := fmt.Sprintf("%s_API_KEY", strings.ToUpper(providerName))
			apiKey := os.Getenv(apiKeyEnv)
			if apiKey == "" {
				t.Fatalf("Missing API key: %s environment variable not set", apiKeyEnv)
			}

			// Create a new LLM instance with proper configuration
			opts := []gollm.ConfigOption{
				gollm.SetProvider(providerName),
				gollm.SetModel(model),
				gollm.SetAPIKey(apiKey),
				gollm.SetMaxRetries(3),
				gollm.SetRetryDelay(time.Second * 2),
				gollm.SetLogLevel(gollm.LogLevelInfo),
				gollm.SetEnableCaching(true),
				gollm.SetTimeout(30 * time.Second),
				gollm.WithStream(true),
			}

			// Add provider-specific settings
			switch providerName {
			case "anthropic":
				opts = append(opts,
					gollm.SetMaxTokens(1000),
				)
			case "openai":
				opts = append(opts,
					gollm.SetMaxTokens(500),
				)
			}

			llm, err := gollm.NewLLM(opts...)
			if err != nil {
				t.Fatalf("Failed to create LLM instance: %v", err)
			}

			// Configure temperature
			llm.SetOption("temperature", 0.7)

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			// Create prompt with system message
			prompt := gollm.NewPrompt(
				"Write a short greeting.",
				gollm.WithSystemPrompt("You are a helpful assistant.", gollm.CacheTypeEphemeral),
			)

			// Test streaming support
			if !llm.SupportsStreaming() {
				t.Fatalf("Provider %s should support streaming", providerName)
			}

			// Get stream
			stream, err := llm.Stream(ctx, prompt)
			if err != nil {
				t.Fatalf("Failed to create stream: %v", err)
			}
			if stream == nil {
				t.Fatal("Stream is nil but no error was returned")
			}
			defer stream.Close()

			// Read first token to verify stream is working
			token, err := stream.Next(ctx)
			if err != nil {
				if err == io.EOF {
					t.Fatal("Stream ended immediately")
				}
				t.Fatalf("Failed to read first token: %v", err)
			}
			if token == nil {
				t.Fatal("First token is nil but no error was returned")
			}
			if token.Text == "" {
				t.Fatal("First token has empty text")
			}

			t.Logf("First token received: %q", token.Text)

			// Read remaining tokens
			var tokens []string
			var fullResponse strings.Builder
			fullResponse.WriteString(token.Text)

			for {
				token, err := stream.Next(ctx)
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Logf("Error reading token: %v", err)
					break
				}
				tokens = append(tokens, token.Text)
				fullResponse.WriteString(token.Text)
			}

			t.Logf("Total tokens received: %d", len(tokens)+1)
			t.Logf("Full response: %q", fullResponse.String())
		})
	}
}
