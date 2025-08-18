package providers

import (
	"encoding/json"
	"github.com/modelcontextprotocol/go-sdk/jsonschema"
)

// Request represents a unified request structure
type Request struct {
	ResponseSchema *jsonschema.Schema `json:"response_schema,omitempty"`
	SystemPrompt   string             `json:"system_prompt,omitempty"`
	Messages       []Message          `json:"messages"`
}

// Message represents a single message in the conversation
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	CacheType  CacheType  `json:"cache_type,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// Response represents the response from an LLM model.
// It contains the content of the response and optional usage information.
type Response struct {
	Content    Content
	Usage      *Usage     `json:"usage,omitempty"`
	Role       string     `json:"role"`
	CacheType  CacheType  `json:"cache_type,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

// Content is a sealed interface for different types of content in a response. Currently, text content only.
type Content interface {
	isContent()
}

// AsText attempts to extract the text content from the response.
func (r *Response) AsText() string {
	if textContent, ok := r.Content.(Text); ok {
		return textContent.Value
	}
	return ""
}

// Text represents text content in a response.
// It implements the Content interface, allowing it to be used as a response content type.
type Text struct {
	Value string
}

// isContent is a method that satisfies the Content interface.
func (t Text) isContent() {}

// CacheType defines how prompts and responses should be cached in the system.
type CacheType string

const (
	// CacheTypeEphemeral indicates that cached responses should only persist
	// for the duration of the program's execution.
	CacheTypeEphemeral CacheType = "ephemeral"
)

// Usage represents the token usage information for a response.
// It includes the number of input tokens, output tokens, cached tokens, and total tokens used in the response.
type Usage struct {
	InputTokens        int64 `json:"input_tokens"`
	CachedInputTokens  int64 `json:"cached_input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	CachedOutputTokens int64 `json:"cached_output_tokens"`
	ReasoningTokens    int64 `json:"reasoning_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
}

func NewUsage(inputTokens, cachedInputTokens, outputTokens, cachedOutputTokens, reasoningTokens int64) *Usage {
	return &Usage{
		InputTokens:        inputTokens,
		CachedInputTokens:  cachedInputTokens,
		OutputTokens:       outputTokens,
		CachedOutputTokens: cachedOutputTokens,
		ReasoningTokens:    reasoningTokens,
		TotalTokens:        (inputTokens - cachedInputTokens) + (outputTokens - cachedOutputTokens),
	}
}

// ToolCall represents a request from the LLM to use a specific tool.
// It includes the tool's identifier, type, and any arguments needed for execution.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}
