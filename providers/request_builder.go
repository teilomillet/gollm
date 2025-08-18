package providers

import "github.com/modelcontextprotocol/go-sdk/jsonschema"

// RequestBuilder helps construct Request objects
type RequestBuilder struct {
	responseSchema *jsonschema.Schema
	systemPrompt   string
	messages       []Message
}

// NewRequestBuilder creates a new request builder
func NewRequestBuilder() *RequestBuilder {
	return &RequestBuilder{
		messages: []Message{},
	}
}

// WithPrompt adds a simple user prompt (backward compatibility)
func (rb *RequestBuilder) WithPrompt(prompt string) *RequestBuilder {
	rb.messages = append(rb.messages, Message{
		Role:    "user",
		Content: prompt,
	})
	return rb
}

// WithMessages adds multiple messages
func (rb *RequestBuilder) WithMessages(messages []Message) *RequestBuilder {
	rb.messages = append(rb.messages, messages...)
	return rb
}

// WithMessage adds a single message
func (rb *RequestBuilder) WithMessage(role, content string) *RequestBuilder {
	rb.messages = append(rb.messages, Message{
		Role:    role,
		Content: content,
	})
	return rb
}

// WithSystemPrompt sets the system prompt
func (rb *RequestBuilder) WithSystemPrompt(prompt string) *RequestBuilder {
	rb.systemPrompt = prompt
	return rb
}

// WithResponseSchema sets the structured response schema
func (rb *RequestBuilder) WithResponseSchema(schema *jsonschema.Schema) *RequestBuilder {
	rb.responseSchema = schema
	return rb
}

// Build creates the final Request object
func (rb *RequestBuilder) Build() *Request {
	return &Request{
		Messages:       rb.messages,
		ResponseSchema: rb.responseSchema,
		SystemPrompt:   rb.systemPrompt,
	}
}
