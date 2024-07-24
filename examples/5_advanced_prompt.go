// File: examples/5_advanced_prompt.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/teilomillet/goal"
)

// AnalysisResult represents the structured output of our analysis
type AnalysisResult struct {
	Perspectives []Perspective `json:"perspectives"`
	Summary      string        `json:"summary"`
}

// Perspective represents a single perspective in the analysis
type Perspective struct {
	Name         string   `json:"name"`
	Implications []string `json:"implications"`
}

func main() {

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance with configuration options
	llmClient, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-4o-mini"),
		goal.SetTemperature(0.7),
		goal.SetMaxTokens(1000),
		goal.SetTimeout(30*time.Second),
		goal.SetMaxRetries(3),
		goal.SetRetryDelay(1*time.Second),
		goal.SetDebugLevel(goal.LogLevelInfo),
		goal.SetAPIKey(apiKey),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Define reusable analysis directives
	balancedAnalysisDirectives := []string{
		"Consider technological, economic, social, and ethical implications",
		"Provide at least one potential positive and one potential negative outcome for each perspective",
		"Ensure the analysis is balanced and objective",
	}

	// Create a custom prompt template for balanced analysis
	balancedAnalysisTemplate := goal.NewPromptTemplate(
		"BalancedAnalysis",
		"Analyze a topic from multiple perspectives",
		`Analyze the following topic from multiple perspectives: {{.Topic}}

Please structure your response as a JSON object with the following format:
{
  "perspectives": [
    {
      "name": "Perspective Name",
      "implications": [
        "Positive implication",
        "Negative implication"
      ]
    }
  ],
  "summary": "A brief, balanced summary of the analysis"
}`,
		goal.WithPromptOptions(
			goal.WithDirectives(balancedAnalysisDirectives...),
			goal.WithMaxLength(500),
		),
	)

	// Define topics for analysis
	topics := []string{
		"The impact of artificial intelligence on job markets",
		"The role of social media in modern democracy",
		"The transition to renewable energy sources",
	}

	// Analyze each topic
	for _, topic := range topics {
		fmt.Printf("Analyzing topic: %s\n", topic)

		// Execute the prompt template
		prompt, err := balancedAnalysisTemplate.Execute(map[string]interface{}{
			"Topic": topic,
		})
		if err != nil {
			log.Fatalf("Failed to execute prompt template for topic '%s': %v", topic, err)
		}

		// Generate the analysis
		analysisJSON, err := llmClient.Generate(ctx, prompt, goal.WithJSONSchemaValidation())
		if err != nil {
			log.Fatalf("Failed to generate analysis for topic '%s': %v", topic, err)
		}

		// Parse the JSON response
		var result AnalysisResult
		err = json.Unmarshal([]byte(analysisJSON), &result)
		if err != nil {
			log.Printf("Warning: Failed to parse analysis JSON for topic '%s': %v", topic, err)
			log.Printf("Raw response: %s", analysisJSON)
			continue // Skip to the next topic instead of exiting
		}

		// Print the structured analysis
		fmt.Printf("Analysis for topic: %s\n", topic)
		for _, perspective := range result.Perspectives {
			fmt.Printf("  Perspective: %s\n", perspective.Name)
			for _, implication := range perspective.Implications {
				fmt.Printf("    - %s\n", implication)
			}
		}
		fmt.Printf("  Summary: %s\n\n", result.Summary)

		// Demonstrate additional goal package features
		summary, err := goal.Summarize(ctx, llmClient, analysisJSON, goal.WithMaxLength(50))
		if err != nil {
			log.Fatalf("Failed to generate summary: %v", err)
		}
		fmt.Printf("Brief summary (50 words): %s\n", summary)

		keyPoints, err := goal.ChainOfThought(ctx, llmClient, fmt.Sprintf("Extract 3 key points from this analysis:\n%s", analysisJSON))
		if err != nil {
			log.Fatalf("Failed to extract key points: %v", err)
		}
		fmt.Printf("Key points:\n%s\n\n", keyPoints)
	}

	// Demonstrate error handling and retries
	_, err = goal.QuestionAnswer(ctx, llmClient, "This is an intentionally long prompt that exceeds the token limit to demonstrate error handling.")
	if err != nil {
		fmt.Printf("Expected error occurred: %v\n", err)
		// Here you would typically implement appropriate error handling or fallback strategies
	}
}
