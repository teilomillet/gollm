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

	// Create LLM client with settings similar to the example
	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama3-8b-8192"),
		gollm.SetMaxTokens(300), // Match the example's token limit
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetTimeout(30*time.Second),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(5*time.Second),
		gollm.SetAPIKey(os.Getenv("GROQ_API_KEY")),
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
				gollm.WithMaxLength(300),
				gollm.WithContext("Climate change is causing global temperature increases and changing precipitation patterns."),
				gollm.WithExamples("Effect: Shifting growing seasons, Adaptation: Developing heat-resistant crops"),
				gollm.WithDirectives(
					"Break down the problem into steps",
					"Show your reasoning for each step",
				),
			},
			validate: func(t *testing.T, response string) {
				// Validate response structure
				assert.True(t, strings.Contains(response, "Chain of Thought:"), "Should include template output")

				// Validate step-by-step reasoning
				assert.True(t, strings.Contains(response, "1.") ||
					strings.Contains(response, "First") ||
					strings.Contains(response, "Step 1"),
					"Should show step numbering")

				// Validate context integration
				assert.True(t, strings.Contains(response, "temperature") ||
					strings.Contains(response, "Temperature"),
					"Should incorporate context about temperature")

				// Validate example integration
				assert.True(t, strings.Contains(response, "season") ||
					strings.Contains(response, "Season") ||
					strings.Contains(response, "crop") ||
					strings.Contains(response, "Crop"),
					"Should incorporate examples")
			},
		},
		{
			name:     "Test Context Only",
			question: "What are the effects of deforestation?",
			opts: []gollm.PromptOption{
				gollm.WithContext("Deforestation is the large-scale removal of forest areas."),
			},
			validate: func(t *testing.T, response string) {
				assert.True(t, strings.Contains(response, "forest") ||
					strings.Contains(response, "Forest"),
					"Should incorporate context")
			},
		},
		{
			name:     "Test Examples Only",
			question: "How to improve study habits?",
			opts: []gollm.PromptOption{
				gollm.WithExamples("Technique: Pomodoro method, Benefit: Improved focus"),
			},
			validate: func(t *testing.T, response string) {
				assert.True(t, strings.Contains(response, "technique") ||
					strings.Contains(response, "Technique") ||
					strings.Contains(response, "method") ||
					strings.Contains(response, "Method"),
					"Should incorporate examples")
			},
		},
		{
			name:     "Test Custom Directives",
			question: "Explain photosynthesis.",
			opts: []gollm.PromptOption{
				gollm.WithDirectives(
					"Focus on chemical reactions",
					"Explain energy transformation",
				),
			},
			validate: func(t *testing.T, response string) {
				assert.True(t, strings.Contains(response, "chemical") ||
					strings.Contains(response, "Chemical") ||
					strings.Contains(response, "energy") ||
					strings.Contains(response, "Energy"),
					"Should follow custom directives")
			},
		},
		{
			name:     "Error Case - Empty Question",
			question: "",
			wantErr:  true,
		},
		{
			name:     "Error Case - Nil Context",
			question: "What is 2 + 2?",
			wantErr:  true,
			validate: func(t *testing.T, response string) {
				ctx := context.Background()
				ctx = nil
				_, err := presets.ChainOfThought(ctx, llm, "What is 2 + 2?")
				assert.Error(t, err, "Should error with nil context")
			},
		},
		{
			name:     "Error Case - Invalid Template",
			question: string([]byte{0xFF, 0xFE, 0xFD}), // Invalid UTF-8
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ctx context.Context
			if tt.name == "Error Case - Nil Context" {
				ctx = nil
			} else {
				ctx = context.Background()
			}

			response, err := presets.ChainOfThought(ctx, llm, tt.question, tt.opts...)

			if tt.wantErr {
				assert.Error(t, err, "Expected error for invalid input")
				return
			}

			assert.NoError(t, err, "Should not return error for valid input")
			assert.NotEmpty(t, response, "Should return non-empty response")

			// Run test-specific validations
			if tt.validate != nil {
				tt.validate(t, response)
			}

			// Log the response for manual review
			// t.Logf("Question: %s\nResponse:\n%s\n", tt.question, response)
		})
	}
}
