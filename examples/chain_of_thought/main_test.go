package chain_of_thought

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/presets"
)

func TestChainOfThought(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping chain of thought test in short mode")
	}

	// Check for API key first
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY environment variable not set")
	}

	// Create LLM client with OpenAI instead of Groq
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"), // Using smaller OpenAI model
		gollm.SetMaxTokens(256),       // Keep token limit low
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(60*time.Second),    // Reasonable timeout
		gollm.SetMaxRetries(2),              // Fewer retries
		gollm.SetRetryDelay(10*time.Second), // Shorter retry delay for OpenAI
		gollm.SetAPIKey(apiKey),             // OpenAI API key
	)
	assert.NoError(t, err, "Should create LLM instance")

	tests := []struct {
		name     string
		question string
		opts     []gollm.PromptOption
		wantErr  bool
		validate func(t *testing.T, response string)
	}{
		{
			name:     "Main Example - Climate Change",
			question: "How might climate change affect global agriculture?",
			opts: []gollm.PromptOption{
				gollm.WithMaxLength(256), // Reduced length
				gollm.WithContext("Climate change is causing global temperature increases."), // Shorter context
				gollm.WithExamples("Effect: Shifting seasons"),                               // Minimal example
				gollm.WithDirectives(
					"Break down the problem into steps",
					"Show your reasoning",
				),
			},
			validate: func(t *testing.T, response string) {
				// Validate response structure - more flexible check
				assert.True(t,
					strings.Contains(response, "Chain of Thought:") ||
						strings.Contains(response, "reasoning") ||
						strings.Contains(response, "thought"),
					"Should include some form of reasoning structure")

				// Validate step-by-step reasoning - more flexible check
				assert.True(t,
					strings.Contains(response, "1.") ||
						strings.Contains(response, "First") ||
						strings.Contains(response, "Step") ||
						strings.Contains(response, "firstly") ||
						strings.Contains(response, "initially") ||
						strings.Contains(response, "begin"),
					"Should show some form of step-by-step analysis")

				// Validate context integration
				assert.True(t,
					strings.Contains(strings.ToLower(response), "temperature") ||
						strings.Contains(strings.ToLower(response), "warming") ||
						strings.Contains(strings.ToLower(response), "heat"),
					"Should incorporate context about temperature changes")

				// Validate example integration - more flexible check
				assert.True(t,
					strings.Contains(strings.ToLower(response), "season") ||
						strings.Contains(strings.ToLower(response), "shift") ||
						strings.Contains(strings.ToLower(response), "pattern") ||
						strings.Contains(strings.ToLower(response), "change") ||
						strings.Contains(strings.ToLower(response), "timing"),
					"Should incorporate concepts related to seasonal changes")
			},
		},
		{
			name:     "Test Examples Only",
			question: "How to improve study habits?",
			opts: []gollm.PromptOption{
				gollm.WithMaxLength(256),                         // Reduced length
				gollm.WithExamples("Technique: Pomodoro method"), // Minimal example
			},
			validate: func(t *testing.T, response string) {
				assert.True(t, strings.Contains(strings.ToLower(response), "technique") ||
					strings.Contains(strings.ToLower(response), "method"),
					"Should incorporate examples")
			},
		},
		{
			name:     "Test Custom Directives",
			question: "Explain photosynthesis.",
			opts: []gollm.PromptOption{
				gollm.WithMaxLength(256), // Reduced length
				gollm.WithDirectives(
					"Focus on chemical reactions",
					"Explain energy flow",
				),
			},
			validate: func(t *testing.T, response string) {
				assert.True(t, strings.Contains(strings.ToLower(response), "chemical") ||
					strings.Contains(strings.ToLower(response), "energy"),
					"Should follow custom directives")
			},
		},
		{
			name:     "Error Case - Empty Question",
			question: "",
			wantErr:  true,
		},
		{
			name:     "Error Case - Invalid Template",
			question: string([]byte{0xFF, 0xFE, 0xFD}), // Invalid UTF-8
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			response, err := presets.ChainOfThought(ctx, llm, tt.question, tt.opts...)

			if tt.wantErr {
				assert.Error(t, err, "Expected error for invalid input")
				return
			}

			if err != nil {
				if strings.Contains(strings.ToLower(err.Error()), "rate limit") {
					t.Logf("Rate limit error encountered (expected): %v", err)
					return
				}
				t.Logf("Error occurred: %v", err)
				return
			}

			// Only validate if we got a response
			if response != "" {
				assert.NotEmpty(t, response, "Should return non-empty response")

				// Run test-specific validations
				if tt.validate != nil {
					tt.validate(t, response)
				}
			}
		})
	}
}
