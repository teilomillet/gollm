package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/utils"
)

// MCPServer represents our test MCP server implementation
type MCPServer struct {
	server *httptest.Server
}

// NewMCPServer creates a new test MCP server
func NewMCPServer(t *testing.T) *MCPServer {
	mux := http.NewServeMux()

	// Initialize server with tool handlers
	mux.HandleFunc("/tools/get_weather", func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Location string `json:"location"`
			Unit     string `json:"unit"`
		}
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		response := WeatherResponse{
			Location:    request.Location,
			Temperature: 22.5,
			Unit:        request.Unit,
			Conditions:  "Partly cloudy",
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	mux.HandleFunc("/tools/get_time", func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Location string `json:"location"`
			Timezone string `json:"timezone"`
		}
		err := json.NewDecoder(r.Body).Decode(&request)
		require.NoError(t, err)

		response := TimeResponse{
			Location: request.Location,
			Time:     time.Now(),
			Timezone: request.Timezone,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	// Initialize server with tool definitions
	mux.HandleFunc("/mcp/list_tools", func(w http.ResponseWriter, r *http.Request) {
		tools := []utils.Tool{
			{
				Type: "function",
				Function: utils.Function{
					Name:        "get_weather",
					Description: "Get the current weather for a location",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
							"unit": map[string]interface{}{
								"type":        "string",
								"enum":        []string{"celsius", "fahrenheit"},
								"description": "The unit for the temperature",
							},
						},
						"required": []string{"location"},
					},
				},
			},
			{
				Type: "function",
				Function: utils.Function{
					Name:        "get_time",
					Description: "Get the current time for a location",
					Parameters: map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state, e.g. San Francisco, CA",
							},
							"timezone": map[string]interface{}{
								"type":        "string",
								"description": "The timezone to use (optional)",
							},
						},
						"required": []string{"location"},
					},
				},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools": tools,
		})
	})

	server := httptest.NewServer(mux)
	return &MCPServer{server: server}
}

// Close shuts down the test server
func (s *MCPServer) Close() {
	s.server.Close()
}

func TestMCPServerIntegration(t *testing.T) {
	// Start MCP server
	mcpServer := NewMCPServer(t)
	defer mcpServer.Close()

	// Create LLM client
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetMaxTokens(256),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	require.NoError(t, err)

	// Test cases
	testCases := []struct {
		name     string
		query    string
		validate func(t *testing.T, response string)
	}{
		{
			name:  "Weather Query",
			query: "What's the weather like in Tokyo?",
			validate: func(t *testing.T, response string) {
				assert.Contains(t, response, "get_weather")
				assert.Contains(t, response, "Tokyo")
			},
		},
		{
			name:  "Time Query",
			query: "What time is it in London?",
			validate: func(t *testing.T, response string) {
				assert.Contains(t, response, "get_time")
				assert.Contains(t, response, "London")
			},
		},
		{
			name:  "Combined Query",
			query: "What's the weather and time in Paris?",
			validate: func(t *testing.T, response string) {
				assert.Contains(t, response, "get_weather")
				assert.Contains(t, response, "get_time")
				assert.Contains(t, response, "Paris")
			},
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create prompt with MCP tools
			prompt := gollm.NewPrompt(
				tc.query,
				gollm.WithTools([]utils.Tool{
					{
						Type: "function",
						Function: utils.Function{
							Name:        "get_weather",
							Description: "Get the current weather for a location",
							Parameters: map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"location": map[string]interface{}{
										"type":        "string",
										"description": "The city and state, e.g. San Francisco, CA",
									},
									"unit": map[string]interface{}{
										"type":        "string",
										"enum":        []string{"celsius", "fahrenheit"},
										"description": "The unit for the temperature",
									},
								},
								"required": []string{"location"},
							},
						},
					},
					{
						Type: "function",
						Function: utils.Function{
							Name:        "get_time",
							Description: "Get the current time for a location",
							Parameters: map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"location": map[string]interface{}{
										"type":        "string",
										"description": "The city and state, e.g. San Francisco, CA",
									},
									"timezone": map[string]interface{}{
										"type":        "string",
										"description": "The timezone to use (optional)",
									},
								},
								"required": []string{"location"},
							},
						},
					},
				}),
			)

			// Generate response
			ctx := context.Background()
			response, err := llm.Generate(ctx, prompt)
			require.NoError(t, err)

			// Validate response
			tc.validate(t, response)
		})
	}
}
