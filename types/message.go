// Package types contains shared type definitions used across the gollm library.
// It helps avoid import cycles while providing common data structures.
package types

// ContentPartType represents the type of content in a multimodal message.
type ContentPartType string

const (
	// ContentTypeText represents text content.
	ContentTypeText ContentPartType = "text"
	// ContentTypeImageURL represents an image referenced by URL.
	ContentTypeImageURL ContentPartType = "image_url"
	// ContentTypeImage represents an image with embedded data (base64).
	ContentTypeImage ContentPartType = "image"
)

// ImageURL represents an image referenced by URL.
// Used by OpenAI and similar providers.
type ImageURL struct {
	URL    string `json:"url"`              // URL of the image or base64 data URI
	Detail string `json:"detail,omitempty"` // Detail level: "auto", "low", or "high"
}

// ImageSource represents an image with embedded data.
// Used by Anthropic and similar providers.
type ImageSource struct {
	Type      string `json:"type"`       // Source type: "base64"
	MediaType string `json:"media_type"` // MIME type: "image/jpeg", "image/png", "image/gif", "image/webp"
	Data      string `json:"data"`       // Base64-encoded image data
}

// ContentPart represents a single part of multimodal content.
// A message can contain multiple parts (e.g., text and images).
type ContentPart struct {
	Type     ContentPartType `json:"type"`                // Type of content: "text", "image_url", or "image"
	Text     string          `json:"text,omitempty"`      // Text content (when Type is "text")
	ImageURL *ImageURL       `json:"image_url,omitempty"` // Image URL (when Type is "image_url")
	Source   *ImageSource    `json:"source,omitempty"`    // Image source (when Type is "image", used by Anthropic)
}

// NewTextContent creates a text content part.
func NewTextContent(text string) ContentPart {
	return ContentPart{
		Type: ContentTypeText,
		Text: text,
	}
}

// NewImageURLContent creates an image content part from a URL.
// The detail parameter can be "auto", "low", or "high" (empty defaults to "auto").
func NewImageURLContent(url string, detail string) ContentPart {
	if detail == "" {
		detail = "auto"
	}
	return ContentPart{
		Type: ContentTypeImageURL,
		ImageURL: &ImageURL{
			URL:    url,
			Detail: detail,
		},
	}
}

// NewImageBase64Content creates an image content part from base64-encoded data.
// mediaType should be "image/jpeg", "image/png", "image/gif", or "image/webp".
func NewImageBase64Content(base64Data, mediaType string) ContentPart {
	return ContentPart{
		Type: ContentTypeImage,
		Source: &ImageSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      base64Data,
		},
	}
}

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
//
// For tool calling support:
// - Assistant messages may include ToolCalls (requests to use tools)
// - Tool messages contain ToolCallID (linking result to the original call)
//
// For multimodal support:
// - Use MultiContent for messages with images or mixed content
// - When MultiContent is set, Content is ignored by providers that support multimodal
type MemoryMessage struct {
	Role         string                 // Role of the message sender (e.g., "user", "assistant", "tool")
	Content      string                 // The actual message content (text-only)
	MultiContent []ContentPart          // Multimodal content (text + images); takes precedence over Content
	Tokens       int                    // Number of tokens in the message
	CacheControl string                 // Caching strategy for this message ("ephemeral", "persistent", etc.)
	Metadata     map[string]interface{} // Additional provider-specific metadata
	ToolCalls    []ToolCall             // Tool calls requested by the assistant (only for role="assistant")
	ToolCallID   string                 // ID of the tool call this message responds to (only for role="tool")
}

// HasMultiContent returns true if the message contains multimodal content.
func (m *MemoryMessage) HasMultiContent() bool {
	return len(m.MultiContent) > 0
}

// GetTextContent returns the text content of the message.
// If MultiContent is set, it concatenates all text parts.
// Otherwise, it returns the Content field.
func (m *MemoryMessage) GetTextContent() string {
	if !m.HasMultiContent() {
		return m.Content
	}
	var text string
	for _, part := range m.MultiContent {
		if part.Type == ContentTypeText {
			text += part.Text
		}
	}
	return text
}
