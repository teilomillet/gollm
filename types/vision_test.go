package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewTextContent(t *testing.T) {
	content := NewTextContent("Hello, world!")

	assert.Equal(t, ContentTypeText, content.Type)
	assert.Equal(t, "Hello, world!", content.Text)
	assert.Nil(t, content.ImageURL)
	assert.Nil(t, content.Source)
}

func TestNewImageURLContent(t *testing.T) {
	t.Run("with default detail", func(t *testing.T) {
		content := NewImageURLContent("https://example.com/image.jpg", "")

		assert.Equal(t, ContentTypeImageURL, content.Type)
		assert.NotNil(t, content.ImageURL)
		assert.Equal(t, "https://example.com/image.jpg", content.ImageURL.URL)
		assert.Equal(t, "auto", content.ImageURL.Detail)
	})

	t.Run("with high detail", func(t *testing.T) {
		content := NewImageURLContent("https://example.com/image.jpg", "high")

		assert.Equal(t, ContentTypeImageURL, content.Type)
		assert.NotNil(t, content.ImageURL)
		assert.Equal(t, "https://example.com/image.jpg", content.ImageURL.URL)
		assert.Equal(t, "high", content.ImageURL.Detail)
	})
}

func TestNewImageBase64Content(t *testing.T) {
	content := NewImageBase64Content("base64data", "image/png")

	assert.Equal(t, ContentTypeImage, content.Type)
	assert.NotNil(t, content.Source)
	assert.Equal(t, "base64", content.Source.Type)
	assert.Equal(t, "image/png", content.Source.MediaType)
	assert.Equal(t, "base64data", content.Source.Data)
}

func TestMemoryMessageMultiContent(t *testing.T) {
	t.Run("HasMultiContent returns false for empty", func(t *testing.T) {
		msg := MemoryMessage{
			Role:    "user",
			Content: "text only",
		}
		assert.False(t, msg.HasMultiContent())
	})

	t.Run("HasMultiContent returns true when set", func(t *testing.T) {
		msg := MemoryMessage{
			Role: "user",
			MultiContent: []ContentPart{
				NewTextContent("Hello"),
				NewImageURLContent("https://example.com/img.jpg", "auto"),
			},
		}
		assert.True(t, msg.HasMultiContent())
	})

	t.Run("GetTextContent returns Content when no MultiContent", func(t *testing.T) {
		msg := MemoryMessage{
			Role:    "user",
			Content: "Hello, world!",
		}
		assert.Equal(t, "Hello, world!", msg.GetTextContent())
	})

	t.Run("GetTextContent concatenates text from MultiContent", func(t *testing.T) {
		msg := MemoryMessage{
			Role: "user",
			MultiContent: []ContentPart{
				NewTextContent("Hello "),
				NewImageURLContent("https://example.com/img.jpg", "auto"),
				NewTextContent("world!"),
			},
		}
		assert.Equal(t, "Hello world!", msg.GetTextContent())
	})
}
