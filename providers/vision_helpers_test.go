package providers

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
)

func TestParseDataURI(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		wantMediaType string
		wantData      string
		wantOK        bool
	}{
		{
			name:          "valid PNG data URI",
			input:         "data:image/png;base64,iVBORw0KGgo=",
			wantMediaType: "image/png",
			wantData:      "iVBORw0KGgo=",
			wantOK:        true,
		},
		{
			name:          "valid JPEG data URI",
			input:         "data:image/jpeg;base64,/9j/4AAQ=",
			wantMediaType: "image/jpeg",
			wantData:      "/9j/4AAQ=",
			wantOK:        true,
		},
		{
			name:          "valid WebP data URI",
			input:         "data:image/webp;base64,UklGR=",
			wantMediaType: "image/webp",
			wantData:      "UklGR=",
			wantOK:        true,
		},
		{
			name:          "valid GIF data URI",
			input:         "data:image/gif;base64,R0lGODlh",
			wantMediaType: "image/gif",
			wantData:      "R0lGODlh",
			wantOK:        true,
		},
		{
			name:          "data URI without base64 encoding specifier",
			input:         "data:text/plain,hello",
			wantMediaType: "text/plain",
			wantData:      "hello",
			wantOK:        true,
		},
		{
			name:   "not a data URI",
			input:  "https://example.com/image.png",
			wantOK: false,
		},
		{
			name:   "empty string",
			input:  "",
			wantOK: false,
		},
		{
			name:   "malformed data URI",
			input:  "data:",
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mediaType, data, ok := ParseDataURI(tt.input)
			assert.Equal(t, tt.wantOK, ok)
			if ok {
				assert.Equal(t, tt.wantMediaType, mediaType)
				assert.Equal(t, tt.wantData, data)
			}
		})
	}
}

func TestContentPartToOpenAIImage(t *testing.T) {
	t.Run("image URL with detail", func(t *testing.T) {
		part := types.ContentPart{
			Type: types.ContentTypeImageURL,
			ImageURL: &types.ImageURL{
				URL:    "https://example.com/image.jpg",
				Detail: "high",
			},
		}
		result, ok := ContentPartToOpenAIImage(part)
		require.True(t, ok, "conversion should succeed")
		assert.Equal(t, "image_url", result["type"])
		imgURL, ok := result["image_url"].(map[string]interface{})
		require.True(t, ok, "image_url should be a map")
		assert.Equal(t, "https://example.com/image.jpg", imgURL["url"])
		assert.Equal(t, "high", imgURL["detail"])
	})

	t.Run("image URL without detail", func(t *testing.T) {
		part := types.ContentPart{
			Type: types.ContentTypeImageURL,
			ImageURL: &types.ImageURL{
				URL: "https://example.com/image.jpg",
			},
		}
		result, ok := ContentPartToOpenAIImage(part)
		require.True(t, ok, "conversion should succeed")
		imgURL, ok := result["image_url"].(map[string]interface{})
		require.True(t, ok, "image_url should be a map")
		assert.Equal(t, "https://example.com/image.jpg", imgURL["url"])
		_, hasDetail := imgURL["detail"]
		assert.False(t, hasDetail)
	})

	t.Run("base64 image", func(t *testing.T) {
		part := types.ContentPart{
			Type: types.ContentTypeImage,
			Source: &types.ImageSource{
				Type:      "base64",
				MediaType: "image/png",
				Data:      "iVBORw0KGgo=",
			},
		}
		result, ok := ContentPartToOpenAIImage(part)
		require.True(t, ok, "conversion should succeed")
		assert.Equal(t, "image_url", result["type"])
		imgURL, ok := result["image_url"].(map[string]interface{})
		require.True(t, ok, "image_url should be a map")
		assert.Equal(t, "data:image/png;base64,iVBORw0KGgo=", imgURL["url"])
	})

	t.Run("nil ImageURL returns false", func(t *testing.T) {
		part := types.ContentPart{
			Type:     types.ContentTypeImageURL,
			ImageURL: nil,
		}
		_, ok := ContentPartToOpenAIImage(part)
		assert.False(t, ok)
	})

	t.Run("nil Source returns false", func(t *testing.T) {
		part := types.ContentPart{
			Type:   types.ContentTypeImage,
			Source: nil,
		}
		_, ok := ContentPartToOpenAIImage(part)
		assert.False(t, ok)
	})
}

func TestContentPartToAnthropicImage(t *testing.T) {
	t.Run("base64 image", func(t *testing.T) {
		part := types.ContentPart{
			Type: types.ContentTypeImage,
			Source: &types.ImageSource{
				Type:      "base64",
				MediaType: "image/png",
				Data:      "iVBORw0KGgo=",
			},
		}
		result, ok := ContentPartToAnthropicImage(part)
		require.True(t, ok, "conversion should succeed")
		assert.Equal(t, "image", result["type"])
		source, ok := result["source"].(map[string]interface{})
		require.True(t, ok, "source should be a map")
		assert.Equal(t, "base64", source["type"])
		assert.Equal(t, "image/png", source["media_type"])
		assert.Equal(t, "iVBORw0KGgo=", source["data"])
	})

	t.Run("URL image", func(t *testing.T) {
		part := types.ContentPart{
			Type: types.ContentTypeImageURL,
			ImageURL: &types.ImageURL{
				URL: "https://example.com/image.jpg",
			},
		}
		result, ok := ContentPartToAnthropicImage(part)
		require.True(t, ok, "conversion should succeed")
		assert.Equal(t, "image", result["type"])
		source, ok := result["source"].(map[string]interface{})
		require.True(t, ok, "source should be a map")
		assert.Equal(t, "url", source["type"])
		assert.Equal(t, "https://example.com/image.jpg", source["url"])
	})

	t.Run("data URI converted to base64", func(t *testing.T) {
		part := types.ContentPart{
			Type: types.ContentTypeImageURL,
			ImageURL: &types.ImageURL{
				URL: "data:image/jpeg;base64,/9j/4AAQ=",
			},
		}
		result, ok := ContentPartToAnthropicImage(part)
		require.True(t, ok, "conversion should succeed")
		assert.Equal(t, "image", result["type"])
		source, ok := result["source"].(map[string]interface{})
		require.True(t, ok, "source should be a map")
		assert.Equal(t, "base64", source["type"])
		assert.Equal(t, "image/jpeg", source["media_type"])
		assert.Equal(t, "/9j/4AAQ=", source["data"])
	})
}

func TestNormalizeContentArray(t *testing.T) {
	t.Run("string content", func(t *testing.T) {
		result := NormalizeContentArray("hello world")
		assert.Len(t, result, 1)
		assert.Equal(t, "text", result[0]["type"])
		assert.Equal(t, "hello world", result[0]["text"])
	})

	t.Run("[]map[string]interface{} passthrough", func(t *testing.T) {
		input := []map[string]interface{}{
			{"type": "text", "text": "hello"},
		}
		result := NormalizeContentArray(input)
		assert.Equal(t, input, result)
	})

	t.Run("[]interface{} conversion", func(t *testing.T) {
		input := []interface{}{
			map[string]interface{}{"type": "text", "text": "hello"},
			map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "http://example.com"}},
		}
		result := NormalizeContentArray(input)
		assert.Len(t, result, 2)
		assert.Equal(t, "text", result[0]["type"])
		assert.Equal(t, "image_url", result[1]["type"])
	})

	t.Run("nil returns empty slice", func(t *testing.T) {
		result := NormalizeContentArray(nil)
		assert.NotNil(t, result)
		assert.Len(t, result, 0)
	})

	t.Run("unknown type returns empty slice", func(t *testing.T) {
		result := NormalizeContentArray(12345)
		assert.NotNil(t, result)
		assert.Len(t, result, 0)
	})
}

func TestBuildOpenAIContentFromParts(t *testing.T) {
	t.Run("mixed text and images", func(t *testing.T) {
		parts := []types.ContentPart{
			{Type: types.ContentTypeText, Text: "Hello"},
			{Type: types.ContentTypeImageURL, ImageURL: &types.ImageURL{URL: "https://example.com/img.jpg", Detail: "high"}},
			{Type: types.ContentTypeText, Text: "World"},
		}
		result := BuildOpenAIContentFromParts(parts)
		require.Len(t, result, 3)
		assert.Equal(t, "text", result[0]["type"])
		assert.Equal(t, "Hello", result[0]["text"])
		assert.Equal(t, "image_url", result[1]["type"])
		assert.Equal(t, "text", result[2]["type"])
		assert.Equal(t, "World", result[2]["text"])
	})

	t.Run("skips invalid image parts", func(t *testing.T) {
		parts := []types.ContentPart{
			{Type: types.ContentTypeText, Text: "Hello"},
			{Type: types.ContentTypeImageURL, ImageURL: nil}, // invalid
			{Type: types.ContentTypeText, Text: "World"},
		}
		result := BuildOpenAIContentFromParts(parts)
		require.Len(t, result, 2)
		assert.Equal(t, "Hello", result[0]["text"])
		assert.Equal(t, "World", result[1]["text"])
	})
}

func TestBuildAnthropicContentFromParts(t *testing.T) {
	t.Run("mixed text and images", func(t *testing.T) {
		parts := []types.ContentPart{
			{Type: types.ContentTypeText, Text: "Describe this:"},
			{Type: types.ContentTypeImage, Source: &types.ImageSource{Type: "base64", MediaType: "image/png", Data: "abc123"}},
		}
		result := BuildAnthropicContentFromParts(parts)
		require.Len(t, result, 2)
		assert.Equal(t, "text", result[0]["type"])
		assert.Equal(t, "Describe this:", result[0]["text"])
		assert.Equal(t, "image", result[1]["type"])
	})
}
