// File: examples/mcp_integration/main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/utils"
)

func Example_MCPClient() error {
	// Create an LLM client
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4"),
		gollm.SetMaxTokens(1024),
		gollm.SetAPIKey(apiKey),
	)
	if err != nil {
		return fmt.Errorf("failed to create LLM: %w", err)
	}

	// Define the tools
	weatherTools := []utils.Tool{
		{
			Type: "function",
			Function: utils.Function{
				Name:        "get_forecast",
				Description: "Get weather forecast for a location using latitude and longitude",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"latitude": map[string]interface{}{
							"type":        "number",
							"description": "Latitude of the location",
						},
						"longitude": map[string]interface{}{
							"type":        "number",
							"description": "Longitude of the location",
						},
					},
					"required": []string{"latitude", "longitude"},
				},
			},
		},
		{
			Type: "function",
			Function: utils.Function{
				Name:        "get_alerts",
				Description: "Get weather alerts for a US state",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"state": map[string]interface{}{
							"type":        "string",
							"description": "Two-letter US state code (e.g. CA, NY)",
						},
					},
					"required": []string{"state"},
				},
			},
		},
	}

	// Query for Paris weather
	prompt := gollm.NewPrompt(
		"What's the weather like in Paris (coordinates: 48.8566, 2.3522)?",
		gollm.WithTools(weatherTools),
		gollm.WithToolChoice("any"),
	)

	fmt.Println("\nChecking Paris weather...")
	response, err := llm.Generate(context.Background(), prompt)
	if err != nil {
		return fmt.Errorf("failed to get Paris weather: %w", err)
	}

	fmt.Printf("\nResponse:\n%s\n", response)

	// Extract and display tool calls
	calls, err := utils.ExtractFunctionCalls(response)
	if err == nil && len(calls) > 0 {
		for i, call := range calls {
			fmt.Printf("\nTool Call %d:\n", i+1)
			fmt.Printf("- Tool: %s\n", call["name"])
			fmt.Printf("- Arguments: %v\n", call["arguments"])

			// Convert the call to JSON
			callJSON, err := json.Marshal(map[string]interface{}{
				"name":      call["name"],
				"arguments": call["arguments"],
			})
			if err != nil {
				fmt.Printf("Error marshaling call: %v\n", err)
				continue
			}

			// Send the call to the MCP server and read the response
			cmd := exec.Command("python3", "-c", fmt.Sprintf(`
import sys, json
call = json.loads('''%s''')
json.dump(call, sys.stdout)
sys.stdout.write("\n")
sys.stdout.flush()
response = sys.stdin.readline().strip()
if response:
    result = json.loads(response)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
sys.stdout.flush()
`, string(callJSON)))

			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr

			if err := cmd.Run(); err != nil {
				fmt.Printf("Error executing tool call: %v\n", err)
				continue
			}

			fmt.Printf("\nTool Response:\n")
		}
	}

	return nil
}

func main() {
	fmt.Println("Running Weather Information Demo")
	if err := Example_MCPClient(); err != nil {
		log.Fatalf("Demo failed: %v\n", err)
	}
}
