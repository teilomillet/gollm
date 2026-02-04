// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/teilomillet/gollm/types"
)

// dataURIPattern matches data URIs like "data:image/png;base64,..."
var dataURIPattern = regexp.MustCompile(`^data:([^;,]+)(?:;[^,]*)?,(.*)$`)

// ParseDataURI extracts the media type and data from a data URI.
// Returns mediaType, data, and whether parsing was successful.
// Example: "data:image/png;base64,iVBORw0..." returns "image/png", "iVBORw0...", true
func ParseDataURI(dataURI string) (mediaType, data string, ok bool) {
	if !strings.HasPrefix(dataURI, "data:") {
		return "", "", false
	}

	matches := dataURIPattern.FindStringSubmatch(dataURI)
	if len(matches) < 3 {
		return "", "", false
	}

	return matches[1], matches[2], true
}

// ContentPartToOpenAIImage converts a ContentPart to OpenAI-style image_url format.
// Returns the formatted map and whether conversion was successful.
func ContentPartToOpenAIImage(part types.ContentPart) (map[string]interface{}, bool) {
	switch part.Type {
	case types.ContentTypeImageURL:
		if part.ImageURL == nil {
			return nil, false
		}
		imgContent := map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url": part.ImageURL.URL,
			},
		}
		if part.ImageURL.Detail != "" {
			imgContent["image_url"].(map[string]interface{})["detail"] = part.ImageURL.Detail
		}
		return imgContent, true

	case types.ContentTypeImage:
		if part.Source == nil || part.Source.Data == "" {
			return nil, false
		}
		// Convert base64 to data URI for OpenAI-compatible format
		dataURI := fmt.Sprintf("data:%s;base64,%s", part.Source.MediaType, part.Source.Data)
		return map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]interface{}{
				"url": dataURI,
			},
		}, true

	default:
		return nil, false
	}
}

// ContentPartToAnthropicImage converts a ContentPart to Anthropic-style image format.
// Returns the formatted map and whether conversion was successful.
func ContentPartToAnthropicImage(part types.ContentPart) (map[string]interface{}, bool) {
	switch part.Type {
	case types.ContentTypeImage:
		// Direct base64 image
		if part.Source == nil || part.Source.Data == "" {
			return nil, false
		}
		return map[string]interface{}{
			"type": "image",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": part.Source.MediaType,
				"data":       part.Source.Data,
			},
		}, true

	case types.ContentTypeImageURL:
		if part.ImageURL == nil {
			return nil, false
		}
		// Check if it's a data URI (base64)
		if mediaType, data, ok := ParseDataURI(part.ImageURL.URL); ok {
			return map[string]interface{}{
				"type": "image",
				"source": map[string]interface{}{
					"type":       "base64",
					"media_type": mediaType,
					"data":       data,
				},
			}, true
		}
		// Regular URL - Anthropic supports URL source type
		return map[string]interface{}{
			"type": "image",
			"source": map[string]interface{}{
				"type": "url",
				"url":  part.ImageURL.URL,
			},
		}, true

	default:
		return nil, false
	}
}

// ConvertImagesToOpenAIContent converts a slice of ContentPart images to OpenAI format.
// Returns a slice of formatted image objects.
func ConvertImagesToOpenAIContent(images []types.ContentPart) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(images))
	for _, img := range images {
		if converted, ok := ContentPartToOpenAIImage(img); ok {
			result = append(result, converted)
		}
	}
	return result
}

// ConvertImagesToAnthropicContent converts a slice of ContentPart images to Anthropic format.
// Returns a slice of formatted image objects.
func ConvertImagesToAnthropicContent(images []types.ContentPart) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(images))
	for _, img := range images {
		if converted, ok := ContentPartToAnthropicImage(img); ok {
			result = append(result, converted)
		}
	}
	return result
}

// NormalizeContentArray safely converts various content representations to []map[string]interface{}.
// Handles: string, []map[string]interface{}, []interface{}, and nil.
func NormalizeContentArray(content interface{}) []map[string]interface{} {
	if content == nil {
		return []map[string]interface{}{}
	}

	switch c := content.(type) {
	case string:
		return []map[string]interface{}{
			{"type": "text", "text": c},
		}
	case []map[string]interface{}:
		return c
	case []interface{}:
		result := make([]map[string]interface{}, 0, len(c))
		for _, item := range c {
			if m, ok := item.(map[string]interface{}); ok {
				result = append(result, m)
			}
		}
		return result
	default:
		return []map[string]interface{}{}
	}
}
