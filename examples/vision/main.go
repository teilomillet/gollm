// Package main demonstrates how to use GoLLM with vision-capable models.
// This example shows how to send images to models like gpt-4o, claude-sonnet-4, llava, etc.
package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	ctx := context.Background()

	// Example 1: Using OpenAI gpt-4o with an image URL
	fmt.Println("\n=== Example 1: OpenAI gpt-4o with Image URL ===")
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetAPIKey(apiKey),
			gollm.SetModel("gpt-4o"), // or "gpt-4-turbo"
			gollm.SetMaxTokens(1000),
		)
		if err != nil {
			fmt.Printf("Error creating OpenAI LLM: %v\n", err)
		} else {
			// Create a prompt with an image URL
			prompt := gollm.NewPrompt(
				"What's in this image? Please describe it in detail.",
				gollm.WithImageURL(
					"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
					"auto", // detail level: "auto", "low", or "high"
				),
			)

			response, err := llm.Generate(ctx, prompt)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Response: %s\n", response)
			}
		}
	} else {
		fmt.Println("Skipping: OPENAI_API_KEY not set")
	}

	// Example 2: Using Anthropic Claude with an image URL
	fmt.Println("\n=== Example 2: Anthropic Claude with Image URL ===")
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("anthropic"),
			gollm.SetAPIKey(apiKey),
			gollm.SetModel("claude-sonnet-4-20250514"), // or any vision-capable Claude model
			gollm.SetMaxTokens(1000),
		)
		if err != nil {
			fmt.Printf("Error creating Anthropic LLM: %v\n", err)
		} else {
			prompt := gollm.NewPrompt(
				"Describe this image in detail.",
				gollm.WithImageURL(
					"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
					"auto",
				),
			)

			response, err := llm.Generate(ctx, prompt)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Response: %s\n", response)
			}
		}
	} else {
		fmt.Println("Skipping: ANTHROPIC_API_KEY not set")
	}

	// Example 3: Using a base64-encoded image
	fmt.Println("\n=== Example 3: Using Base64-encoded Image ===")
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetAPIKey(apiKey),
			gollm.SetModel("gpt-4o"),
			gollm.SetMaxTokens(500),
		)
		if err != nil {
			fmt.Printf("Error creating LLM: %v\n", err)
		} else {
			// In a real application, you would read the image from a file:
			// imageData, err := os.ReadFile("path/to/image.png")
			// base64Data := base64.StdEncoding.EncodeToString(imageData)

			// For this example, we'll create a tiny 1x1 pixel PNG
			// This is a minimal valid PNG (red pixel)
			tinyPNG := []byte{
				0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
				0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
				0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
				0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
				0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
				0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
				0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xFE,
				0xD4, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, // IEND chunk
				0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
			}
			base64Data := base64.StdEncoding.EncodeToString(tinyPNG)

			prompt := gollm.NewPrompt(
				"What color is this single pixel image?",
				gollm.WithImageBase64(base64Data, "image/png"),
			)

			response, err := llm.Generate(ctx, prompt)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Response: %s\n", response)
			}
		}
	} else {
		fmt.Println("Skipping: OPENAI_API_KEY not set")
	}

	// Example 4: Multiple images in a single prompt
	fmt.Println("\n=== Example 4: Multiple Images ===")
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetAPIKey(apiKey),
			gollm.SetModel("gpt-4o"),
			gollm.SetMaxTokens(1000),
		)
		if err != nil {
			fmt.Printf("Error creating LLM: %v\n", err)
		} else {
			prompt := gollm.NewPrompt(
				"Compare these two images. What are the similarities and differences?",
				gollm.WithImageURL("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png", "auto"),
				gollm.WithImageURL("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg", "auto"),
			)

			response, err := llm.Generate(ctx, prompt)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Response: %s\n", response)
			}
		}
	} else {
		fmt.Println("Skipping: OPENAI_API_KEY not set")
	}

	// Example 5: Using OpenRouter with vision models
	fmt.Println("\n=== Example 5: OpenRouter with Vision Model ===")
	if apiKey := os.Getenv("OPENROUTER_API_KEY"); apiKey != "" {
		llm, err := gollm.NewLLM(
			gollm.SetProvider("openrouter"),
			gollm.SetAPIKey(apiKey),
			gollm.SetModel("anthropic/claude-3-5-sonnet"), // Vision-capable model via OpenRouter
			gollm.SetMaxTokens(1000),
		)
		if err != nil {
			fmt.Printf("Error creating OpenRouter LLM: %v\n", err)
		} else {
			prompt := gollm.NewPrompt(
				"What do you see in this image?",
				gollm.WithImageURL(
					"https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png",
					"auto",
				),
			)

			response, err := llm.Generate(ctx, prompt)
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Response: %s\n", response)
			}
		}
	} else {
		fmt.Println("Skipping: OPENROUTER_API_KEY not set")
	}

	// Example 6: Using Ollama with vision models (like llava, bakllava)
	// Note: Ollama requires base64-encoded images, not URLs
	fmt.Println("\n=== Example 6: Ollama with Vision Model (llava) ===")
	fmt.Println("Note: Ollama vision models require base64-encoded images")
	// Check if Ollama is running by attempting to create an LLM
	llm, err := gollm.NewLLM(
		gollm.SetProvider("ollama"),
		gollm.SetModel("llava"), // or "bakllava", "llava:13b", etc.
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		fmt.Printf("Skipping: Could not connect to Ollama: %v\n", err)
	} else {
		// Use the same tiny PNG from Example 3
		tinyPNG := []byte{
			0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
			0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
			0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
			0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
			0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
			0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
			0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xFE,
			0xD4, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
			0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
		}
		base64Data := base64.StdEncoding.EncodeToString(tinyPNG)

		prompt := gollm.NewPrompt(
			"Describe this image. What color is the pixel?",
			gollm.WithImageBase64(base64Data, "image/png"),
		)

		response, err := llm.Generate(ctx, prompt)
		if err != nil {
			fmt.Printf("Error (is llava model installed?): %v\n", err)
		} else {
			fmt.Printf("Response: %s\n", response)
		}
	}

	fmt.Println("\n=== Vision Examples Complete ===")
}
