package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/gollm"
)

func main() {
	llmClient, err := gollm.NewLLM()
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	text := `Artificial intelligence (AI) is transforming various sectors of society, including healthcare, 
	finance, and transportation. While AI offers numerous benefits such as improved efficiency and 
	decision-making, it also raises concerns about privacy, job displacement, and ethical considerations. 
	As AI continues to advance, it's crucial to address these challenges and ensure responsible development 
	and deployment of AI technologies.`

	summary, err := gollm.Summarize(ctx, llmClient, text,
		gollm.WithMaxLength(50),
		gollm.WithDirectives(
			"Provide a concise summary",
			"Capture the main points and key details",
			"Focus on the main impacts and challenges",
		),
	)
	if err != nil {
		log.Fatalf("Summarize failed: %v", err)
	}

	fmt.Printf("Summary:\n%s\n", summary)
}
