package llm

import (
	"encoding/json"

	"github.com/teilomillet/gollm/types"
)

// Response represents a structured response from an LLM.
// It contains the text content, any tool calls, and usage information.
type Response struct {
	Content   string            // The text content of the response
	ToolCalls []types.ToolCall  // Tool calls requested by the LLM (if any)
	Usage     *Usage            // Token usage information (if available)
}

// Usage contains token usage statistics for a response.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// HasToolCalls returns true if the response contains tool calls.
func (r *Response) HasToolCalls() bool {
	return len(r.ToolCalls) > 0
}

// GetToolCall returns the first tool call with the given name, or nil if not found.
func (r *Response) GetToolCall(name string) *types.ToolCall {
	for i := range r.ToolCalls {
		if r.ToolCalls[i].Function.Name == name {
			return &r.ToolCalls[i]
		}
	}
	return nil
}

// GetAllToolCalls returns all tool calls in the response.
func (r *Response) GetAllToolCalls() []types.ToolCall {
	return r.ToolCalls
}

// String returns the text content of the response.
// This allows Response to be used where a string is expected.
func (r *Response) String() string {
	return r.Content
}

// ParseOpenAIToolCalls parses tool calls from an OpenAI-style response.
// This is used internally by providers to convert their response format.
func ParseOpenAIToolCalls(toolCalls []struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}) []types.ToolCall {
	result := make([]types.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = types.ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
		}
		result[i].Function.Name = tc.Function.Name
		result[i].Function.Arguments = tc.Function.Arguments
	}
	return result
}

// ParseAnthropicToolCalls parses tool calls from an Anthropic-style response.
// Anthropic uses "tool_use" content blocks instead of tool_calls array.
func ParseAnthropicToolCalls(contentBlocks []struct {
	Type  string          `json:"type"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}) []types.ToolCall {
	var result []types.ToolCall
	for _, block := range contentBlocks {
		if block.Type == "tool_use" {
			tc := types.ToolCall{
				ID:   block.ID,
				Type: "function",
			}
			tc.Function.Name = block.Name
			tc.Function.Arguments = block.Input
			result = append(result, tc)
		}
	}
	return result
}
