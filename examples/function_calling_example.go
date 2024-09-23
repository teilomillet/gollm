package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	fmt.Println("Starting the function calling example...")
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatalf("ANTHROPIC_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"), // Use a valid model name
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxTokens(300),
		gollm.SetMaxRetries(3),
		gollm.SetDebugLevel(gollm.LogLevelDebug),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Define functions
	getWeatherFunction := gollm.Function{
		Name:        "get_weather",
		Description: "Get the current weather in a given location",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "The city and state, e.g. San Francisco, CA",
				},
				"unit": map[string]interface{}{
					"type": "string",
					"enum": []string{"celsius", "fahrenheit"},
				},
			},
			"required": []string{"location", "unit"},
		},
	}

	// Create a prompt with function calling
	prompt := gollm.NewPrompt(
		"",
		gollm.WithMessages([]gollm.Message{
			{Role: "user", Content: "What's the weather like in New York?"},
		}),
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
	var fullResponse struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	err = json.Unmarshal([]byte(response), &fullResponse)
	if err != nil {
		log.Printf("Failed to unmarshal response: %v", err)
		fmt.Printf("Regular response: %s\n", response)
	} else if len(fullResponse.Choices) > 0 && len(fullResponse.Choices[0].Message.ToolCalls) > 0 {
		fmt.Println("Function call detected:")
		for _, call := range fullResponse.Choices[0].Message.ToolCalls {
			fmt.Printf("Function: %s\n", call.Function.Name)
			fmt.Printf("Raw Arguments: %s\n", call.Function.Arguments)

			// Parse the arguments string into a map
			var args map[string]interface{}
			err = json.Unmarshal([]byte(call.Function.Arguments), &args)
			if err != nil {
				log.Printf("Failed to parse arguments: %v", err)
			} else {
				fmt.Printf("Parsed Arguments: %v\n", args)
			}
		}

		// Simulate function execution
		weatherData := map[string]interface{}{
			"temperature": 22,
			"unit":        "celsius",
			"description": "Partly cloudy",
		}
		weatherResponse, _ := json.Marshal(weatherData)

		// Generate follow-up response
		finalResponse, err := llm.(interface {
			GenerateFunctionCallFollowUp(context.Context, *gollm.Prompt, string, string) (string, error)
		}).GenerateFunctionCallFollowUp(ctx, prompt, response, string(weatherResponse))
		if err != nil {
			log.Fatalf("Failed to generate final response: %v", err)
		}
		fmt.Printf("Final response: %s\n", finalResponse)
	} else {
		fmt.Printf("Regular response: %s\n", response)
	}

	// Example with forced function calling
	fmt.Println("\nExample with forced function calling:")
	forcedPrompt := gollm.NewPrompt(
		"What's the weather like in London?",
		gollm.WithTools([]gollm.Tool{{
			Type:     "function",
			Function: getWeatherFunction,
		}}),
		gollm.WithToolChoice("auto"), // Force the model to call the get_weather function
	)

	forcedResponse, err := llm.Generate(ctx, forcedPrompt)
	if err != nil {
		log.Fatalf("Failed to generate forced response: %v", err)
	}
	fmt.Printf("Forced function call response:\n%s\n", forcedResponse)
}
