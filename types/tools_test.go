package types

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestToolCallGetArguments(t *testing.T) {
	t.Run("valid arguments", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`{"location": "New York", "unit": "celsius"}`)

		args, err := tc.GetArguments()
		assert.NoError(t, err)
		assert.Equal(t, "New York", args["location"])
		assert.Equal(t, "celsius", args["unit"])
	})

	t.Run("invalid JSON arguments", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`invalid json`)

		args, err := tc.GetArguments()
		assert.Error(t, err)
		assert.Nil(t, args)
	})

	t.Run("empty arguments", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`{}`)

		args, err := tc.GetArguments()
		assert.NoError(t, err)
		assert.Empty(t, args)
	})
}

func TestToolCallGetArgumentString(t *testing.T) {
	t.Run("string argument exists", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`{"location": "New York"}`)

		assert.Equal(t, "New York", tc.GetArgumentString("location"))
	})

	t.Run("argument does not exist", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`{"location": "New York"}`)

		assert.Equal(t, "", tc.GetArgumentString("nonexistent"))
	})

	t.Run("argument is not a string", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`{"count": 42}`)

		assert.Equal(t, "", tc.GetArgumentString("count"))
	})

	t.Run("invalid JSON returns empty string", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`invalid`)

		assert.Equal(t, "", tc.GetArgumentString("anything"))
	})
}

func TestToolCallGetName(t *testing.T) {
	tc := ToolCall{
		ID:   "call_123",
		Type: "function",
	}
	tc.Function.Name = "get_weather"
	tc.Function.Arguments = json.RawMessage(`{}`)

	assert.Equal(t, "get_weather", tc.GetName())
}

func TestNewToolResult(t *testing.T) {
	result := NewToolResult("call_123", `{"temperature": 22}`)

	assert.Equal(t, "call_123", result.ToolCallID)
	assert.Equal(t, `{"temperature": 22}`, result.Content)
	assert.False(t, result.IsError)
}

func TestNewToolError(t *testing.T) {
	result := NewToolError("call_123", "API rate limit exceeded")

	assert.Equal(t, "call_123", result.ToolCallID)
	assert.Equal(t, "API rate limit exceeded", result.Content)
	assert.True(t, result.IsError)
}

func TestToolCallJSONSerialization(t *testing.T) {
	t.Run("serialize tool call", func(t *testing.T) {
		tc := ToolCall{
			ID:   "call_123",
			Type: "function",
		}
		tc.Function.Name = "get_weather"
		tc.Function.Arguments = json.RawMessage(`{"location":"NYC"}`)

		data, err := json.Marshal(tc)
		assert.NoError(t, err)

		// Deserialize back
		var tc2 ToolCall
		err = json.Unmarshal(data, &tc2)
		assert.NoError(t, err)
		assert.Equal(t, tc.ID, tc2.ID)
		assert.Equal(t, tc.Type, tc2.Type)
		assert.Equal(t, tc.Function.Name, tc2.Function.Name)
	})
}

func TestToolResultJSONSerialization(t *testing.T) {
	t.Run("serialize success result", func(t *testing.T) {
		result := NewToolResult("call_123", "success")
		data, err := json.Marshal(result)
		assert.NoError(t, err)

		var result2 ToolResult
		err = json.Unmarshal(data, &result2)
		assert.NoError(t, err)
		assert.Equal(t, result.ToolCallID, result2.ToolCallID)
		assert.Equal(t, result.Content, result2.Content)
		assert.False(t, result2.IsError)
	})

	t.Run("serialize error result", func(t *testing.T) {
		result := NewToolError("call_123", "failed")
		data, err := json.Marshal(result)
		assert.NoError(t, err)

		var result2 ToolResult
		err = json.Unmarshal(data, &result2)
		assert.NoError(t, err)
		assert.True(t, result2.IsError)
	})
}
