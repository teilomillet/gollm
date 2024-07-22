package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	cfg := goal.NewConfigBuilder().
		SetProvider("openai").
		SetModel("gpt-3.5-turbo").
		SetMaxTokens(150).
		SetTemperature(0.7).
		SetAPIKey("your-api-key-here"). // Replace with your actual API key
		Build()

	llmClient, err := goal.NewLLM(cfg)
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	text := `Artificial intelligence (AI) is transforming various sectors of society, including healthcare, 
	finance, and transportation. While AI offers numerous benefits such as improved efficiency and 
	decision-making, it also raises concerns about privacy, job displacement, and ethical considerations. 
	As AI continues to advance, it's crucial to address these challenges and ensure responsible development 
	and deployment of AI technologies.`

	summary, err := goal.Summarize(ctx, llmClient, text,
		goal.WithMaxLength(50),
		goal.WithDirectives("Focus on the main impacts and challenges"),
	)
	if err != nil {
		log.Fatalf("Summarize failed: %v", err)
	}

	fmt.Printf("Summary:\n%s\n", summary)
}
