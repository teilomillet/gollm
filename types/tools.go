// Package types contains shared type definitions used across the gollm library.
package types

import "encoding/json"

// ToolCall represents a request from the LLM to use a specific tool.
// This structure matches the OpenAI format which is also used by most providers.
// This is used when the LLM determines it needs to call a function/tool
// to complete its response.
type ToolCall struct {
	ID       string `json:"id"`   // Unique identifier for this tool call
	Type     string `json:"type"` // Type of tool being called (usually "function")
	Function struct {
		Name      string          `json:"name"`      // Name of the function to call
		Arguments json.RawMessage `json:"arguments"` // Arguments for the function call as raw JSON
	} `json:"function"`
}

// GetArguments parses the tool call arguments into a map.
// Returns an error if the arguments cannot be parsed as JSON.
func (tc *ToolCall) GetArguments() (map[string]interface{}, error) {
	var args map[string]interface{}
	if err := json.Unmarshal(tc.Function.Arguments, &args); err != nil {
		return nil, err
	}
	return args, nil
}

// GetArgumentString returns a specific string argument.
// Returns empty string if the argument doesn't exist or is not a string.
func (tc *ToolCall) GetArgumentString(key string) string {
	args, err := tc.GetArguments()
	if err != nil {
		return ""
	}
	if val, ok := args[key].(string); ok {
		return val
	}
	return ""
}

// GetName returns the function name for this tool call.
func (tc *ToolCall) GetName() string {
	return tc.Function.Name
}

// ToolResult represents the result of executing a tool.
// This is used to send the result back to the LLM for it to
// incorporate into its response.
type ToolResult struct {
	ToolCallID string `json:"tool_call_id"` // ID of the tool call this result responds to
	Content    string `json:"content"`      // The result content (typically JSON or text)
	IsError    bool   `json:"is_error,omitempty"` // Whether this result represents an error
}

// NewToolResult creates a successful tool result.
func NewToolResult(toolCallID, content string) ToolResult {
	return ToolResult{
		ToolCallID: toolCallID,
		Content:    content,
		IsError:    false,
	}
}

// NewToolError creates an error tool result.
func NewToolError(toolCallID, errorMessage string) ToolResult {
	return ToolResult{
		ToolCallID: toolCallID,
		Content:    errorMessage,
		IsError:    true,
	}
}
