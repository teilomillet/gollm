package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/mauza/gollm"
	"github.com/mauza/gollm/presets"
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

// cleanJSONResponse removes markdown code block delimiters and trims whitespace
func cleanJSONResponse(response string) string {
	response = strings.TrimSpace(response)
	response = strings.TrimPrefix(response, "```json")
	response = strings.TrimSuffix(response, "```")
	return strings.TrimSpace(response)
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("OPENAI_API_KEY environment variable is not set")
	}

	// Create a new LLM instance with configuration options
	llmClient, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
		gollm.SetTimeout(30*time.Second),
		gollm.SetMaxRetries(3),
		gollm.SetRetryDelay(1*time.Second),
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetAPIKey(apiKey),
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
	balancedAnalysisTemplate := gollm.NewPromptTemplate(
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
		gollm.WithPromptOptions(
			gollm.WithDirectives(balancedAnalysisDirectives...),
			gollm.WithMaxLength(500),
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
		analysisJSON, err := llmClient.Generate(ctx, prompt, gollm.WithJSONSchemaValidation())
		if err != nil {
			log.Fatalf("Failed to generate analysis for topic '%s': %v", topic, err)
		}

		// Clean the JSON response
		cleanedJSON := cleanJSONResponse(analysisJSON)

		// Parse the JSON response
		var result AnalysisResult
		err = json.Unmarshal([]byte(cleanedJSON), &result)
		if err != nil {
			log.Printf("Warning: Failed to parse analysis JSON for topic '%s': %v", topic, err)
			log.Printf("Raw response: %s", cleanedJSON)
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

		// Demonstrate additional gollm package features
		summary, err := presets.Summarize(ctx, llmClient, analysisJSON, gollm.WithMaxLength(50))
		if err != nil {
			log.Fatalf("Failed to generate summary: %v", err)
		}
		fmt.Printf("Brief summary (50 words): %s\n", summary)

		keyPoints, err := presets.ChainOfThought(ctx, llmClient, fmt.Sprintf("Extract 3 key points from this analysis:\n%s", analysisJSON))
		if err != nil {
			log.Fatalf("Failed to extract key points: %v", err)
		}
		fmt.Printf("Key points:\n%s\n\n", keyPoints)
	}

	// Demonstrate error handling and retries
	fmt.Println("\nDemonstrating error handling and retries:")

	// Create a client with short timeout to demonstrate retries
	retryClient, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetTemperature(0.7),
		gollm.SetMaxTokens(1000),
		gollm.SetTimeout(5*time.Second),  // Short timeout to trigger retries
		gollm.SetMaxRetries(3),           // Will attempt 4 times total (initial + 3 retries)
		gollm.SetRetryDelay(time.Second), // Wait between retries
		gollm.SetLogLevel(gollm.LogLevelInfo),
		gollm.SetAPIKey(apiKey),
	)
	if err != nil {
		log.Fatalf("Failed to create retry client: %v", err)
	}

	// Use a complex prompt that's likely to take longer than the timeout
	_, err = presets.QuestionAnswer(ctx, retryClient,
		"Please provide a detailed analysis of the impact of quantum computing on cryptography, including current limitations, potential breakthroughs, and implications for cybersecurity.")

	if err != nil {
		fmt.Printf("Got expected error after retries: %v\n", err)
		fmt.Println("This demonstrates how the retry mechanism handles timeouts and errors gracefully.")
	} else {
		fmt.Println("Request succeeded (unexpected with short timeout)")
	}
}
