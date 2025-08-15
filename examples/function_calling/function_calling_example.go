package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/teilomillet/gollm"
)

func main() {
	fmt.Println("Starting the function calling example...")
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatalf("ANTHROPIC_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-3-5-sonnet-20240620"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetLogLevel(gollm.LogLevelDebug),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Define functions
	getWeatherFunction := gollm.Function{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"location": map[string]any{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
			},
			"required": []string{"location"},
		},
	}

	// Create a prompt with function calling
	prompt := gollm.NewPrompt(
		"What's the weather like in New York?",
		gollm.WithTools([]gollm.Tool{{
			Type:     "function",
			Function: getWeatherFunction,
		}}),
		gollm.WithToolChoice("auto"),
	)

	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	// Parse the response
	if strings.Contains(response.AsText(), "<function_call>") {
		fmt.Println("Function call detected:")
		respText := response.AsText()
		start := strings.Index(respText, "<function_call>") + len("<function_call>")
		end := strings.Index(respText, "</function_call>")
		functionCallJSON := respText[start:end]

		var functionCall struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		json.Unmarshal([]byte(functionCallJSON), &functionCall)

		fmt.Printf("Function: %s\n", functionCall.Name)
		fmt.Printf("Arguments: %s\n", string(functionCall.Arguments))

		// Simulate function execution
		weatherData := map[string]any{
			"temperature": 22,
			"unit":        "celsius",
			"description": "Partly cloudy",
		}
		weatherResponse, _ := json.Marshal(weatherData)

		// Generate follow-up response
		finalResponse, err := llm.Generate(ctx, gollm.NewPrompt(
			fmt.Sprintf("The weather data for New York is: %s. Please provide a human-readable summary.", string(weatherResponse)),
		))
		if err != nil {
			log.Fatalf("Failed to generate final response: %v", err)
		}
		fmt.Printf("Final response: %s\n", finalResponse.AsText())
	} else {
		fmt.Printf("Regular response: %s\n", response.AsText())
	}
}
