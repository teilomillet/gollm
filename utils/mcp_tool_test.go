// File: utils/mcp_tool_test.go
package utils

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolMCPConversion(t *testing.T) {
	tool := &Tool{
		Type: "function",
		Function: Function{
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
	}

	// Test ToMCP conversion
	t.Run("ToMCP", func(t *testing.T) {
		mcp, err := tool.ToMCP()
		require.NoError(t, err)
		assert.Equal(t, tool.Type, mcp.Type)
		assert.Equal(t, tool.Function.Name, mcp.Name)
		assert.Equal(t, tool.Function.Description, mcp.Description)
		assert.Equal(t, tool.Function.Parameters, mcp.Parameters)
	})

	// Test FromMCP conversion
	t.Run("FromMCP", func(t *testing.T) {
		mcp := &MCPTool{
			Type:        tool.Type,
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  tool.Function.Parameters,
		}

		converted, err := FromMCP(mcp)
		require.NoError(t, err)
		assert.Equal(t, tool.Type, converted.Type)
		assert.Equal(t, tool.Function.Name, converted.Function.Name)
		assert.Equal(t, tool.Function.Description, converted.Function.Description)
		assert.Equal(t, tool.Function.Parameters, converted.Function.Parameters)
	})

	// Test JSON marshaling
	t.Run("MarshalJSON", func(t *testing.T) {
		data, err := json.Marshal(tool)
		require.NoError(t, err)

		var mcp MCPTool
		err = json.Unmarshal(data, &mcp)
		require.NoError(t, err)

		assert.Equal(t, tool.Type, mcp.Type)
		assert.Equal(t, tool.Function.Name, mcp.Name)
		assert.Equal(t, tool.Function.Description, mcp.Description)
		assert.Equal(t, tool.Function.Parameters, mcp.Parameters)
	})

	// Test JSON unmarshaling
	t.Run("UnmarshalJSON", func(t *testing.T) {
		mcpJSON := `{
			"type": "function",
			"name": "get_weather",
			"description": "Get the current weather for a location",
			"parameters": {
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. San Francisco, CA"
					},
					"unit": {
						"type": "string",
						"enum": ["celsius", "fahrenheit"],
						"description": "The unit for the temperature"
					}
				},
				"required": ["location"]
			}
		}`

		var converted Tool
		err := json.Unmarshal([]byte(mcpJSON), &converted)
		require.NoError(t, err)

		assert.Equal(t, tool.Type, converted.Type)
		assert.Equal(t, tool.Function.Name, converted.Function.Name)
		assert.Equal(t, tool.Function.Description, converted.Function.Description)
		assert.Equal(t, tool.Function.Parameters, converted.Function.Parameters)
	})

	// Test error cases
	t.Run("ErrorCases", func(t *testing.T) {
		// Test nil MCP tool
		t.Run("NilMCP", func(t *testing.T) {
			_, err := FromMCP(nil)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "cannot convert nil MCP tool")
		})

		// Test invalid JSON
		t.Run("InvalidJSON", func(t *testing.T) {
			var tool Tool
			err := json.Unmarshal([]byte("invalid json"), &tool)
			assert.Error(t, err)
		})

		// Test missing required fields
		t.Run("MissingFields", func(t *testing.T) {
			mcpJSON := `{
				"type": "function"
			}`
			var tool Tool
			err := json.Unmarshal([]byte(mcpJSON), &tool)
			assert.NoError(t, err) // Should not error as fields can be empty
			assert.Equal(t, "function", tool.Type)
			assert.Empty(t, tool.Function.Name)
			assert.Empty(t, tool.Function.Description)
			assert.Empty(t, tool.Function.Parameters)
		})

		// Test empty tool
		t.Run("EmptyTool", func(t *testing.T) {
			tool := &Tool{}
			mcp, err := tool.ToMCP()
			assert.NoError(t, err)
			assert.Empty(t, mcp.Type)
			assert.Empty(t, mcp.Name)
			assert.Empty(t, mcp.Description)
			assert.Empty(t, mcp.Parameters)
		})
	})

	// Test complex parameters
	t.Run("ComplexParameters", func(t *testing.T) {
		complexTool := &Tool{
			Type: "function",
			Function: Function{
				Name:        "process_data",
				Description: "Process complex data structure",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"data": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{
								"type": "object",
								"properties": map[string]interface{}{
									"id": map[string]interface{}{
										"type": "integer",
									},
									"metadata": map[string]interface{}{
										"type":                 "object",
										"additionalProperties": true,
									},
								},
							},
						},
						"options": map[string]interface{}{
							"type":                 "object",
							"additionalProperties": true,
						},
					},
					"required": []string{"data"},
				},
			},
		}

		// Test conversion to MCP and back
		mcp, err := complexTool.ToMCP()
		require.NoError(t, err)
		converted, err := FromMCP(mcp)
		require.NoError(t, err)
		assert.Equal(t, complexTool.Type, converted.Type)
		assert.Equal(t, complexTool.Function.Name, converted.Function.Name)
		assert.Equal(t, complexTool.Function.Description, converted.Function.Description)
		assert.Equal(t, complexTool.Function.Parameters, converted.Function.Parameters)

		// Test JSON roundtrip
		data, err := json.Marshal(complexTool)
		require.NoError(t, err)
		var roundTripped Tool
		err = json.Unmarshal(data, &roundTripped)
		require.NoError(t, err)
		assert.Equal(t, complexTool.Type, roundTripped.Type)
		assert.Equal(t, complexTool.Function.Name, roundTripped.Function.Name)
		assert.Equal(t, complexTool.Function.Description, roundTripped.Function.Description)
		assert.Equal(t, complexTool.Function.Parameters, roundTripped.Function.Parameters)
	})
}
