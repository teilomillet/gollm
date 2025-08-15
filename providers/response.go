package providers

// Content is a sealed interface for different types of content in a response. Currently, text content only.
type Content interface {
	isContent()
}

// Response represents the response from an LLM model.
// It contains the content of the response and optional usage information.
// The content can be of different types, such as text.
// The Usage field provides information about token usage, including input, output, and total tokens
type Response struct {
	Content Content
	Usage   *Usage
}

func (r Response) String() string {
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

// Usage represents the token usage information for a response.
// It includes the number of input tokens, output tokens, cached tokens, and total tokens used in the response.
type Usage struct {
	InputTokens        int64 // Total input tokens used, including cached and non-cached
	CachedInputTokens  int64 // Total cached input tokens used, including write operations
	OutputTokens       int64 // Total output tokens generated, including reasoning and text
	CachedOutputTokens int64 // Total cached output tokens used, including reasoning and text
	TotalTokens        int64 // Total tokens used, which is the sum of input and output (excludes cached tokens).
}

func NewUsage(inputTokens, cachedInputTokens, outputTokens, cachedOutputTokens int64) *Usage {
	return &Usage{
		InputTokens:        inputTokens,
		CachedInputTokens:  cachedInputTokens,
		OutputTokens:       outputTokens,
		CachedOutputTokens: cachedOutputTokens,
		TotalTokens:        (inputTokens - cachedInputTokens) + (outputTokens - cachedOutputTokens),
	}
}
