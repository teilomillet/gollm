// File: examples/5_advanced_prompt.go

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/goal"
)

func main() {
	llmClient, err := goal.NewLLM("")
	if err != nil {
		log.Fatalf("Failed to create LLM client: %v", err)
	}

	ctx := context.Background()

	// Easy: Using a pre-defined high-level function
	easyQuestion := "What are the main benefits of artificial intelligence?"
	easyAnswer, err := goal.QuestionAnswer(ctx, llmClient, easyQuestion, "")
	if err != nil {
		log.Fatalf("Failed to generate easy answer: %v", err)
	}

	fmt.Printf("Easy Question: %s\n", easyQuestion)
	fmt.Printf("Easy Answer:\n%s\n\n", easyAnswer)

	// Advanced: Creating a custom, reusable prompt template
	advancedPromptTemplate := goal.NewPrompt("Analyze the following topic from multiple perspectives:").
		WithDirective("Consider technological, economic, social, and ethical implications").
		WithDirective("Provide at least one potential positive and one potential negative outcome for each perspective").
		WithDirective("Conclude with a balanced summary of no more than 3 sentences").
		WithOutput("Multi-perspective Analysis:")

	// Using the custom prompt template for different topics
	topics := []string{
		"The widespread adoption of artificial intelligence in healthcare",
		"The implementation of a universal basic income",
		"The transition to renewable energy sources",
	}

	for _, topic := range topics {
		fullPrompt := advancedPromptTemplate.WithInput(topic)
		analysis, _, err := llmClient.Generate(ctx, fullPrompt.String())
		if err != nil {
			log.Fatalf("Failed to generate analysis for topic '%s': %v", topic, err)
		}

		fmt.Printf("Topic: %s\n", topic)
		fmt.Printf("Analysis:\n%s\n\n", analysis)
	}

	// Expert: Combining custom prompts with other goal package features
	expertTopic := "The impact of social media on democratic processes"
	expertPrompt := advancedPromptTemplate.WithInput(expertTopic)

	expertAnalysis, _, err := llmClient.Generate(ctx, expertPrompt.String())
	if err != nil {
		log.Fatalf("Failed to generate expert analysis: %v", err)
	}

	summary, err := goal.Summarize(ctx, llmClient, expertAnalysis, 50)
	if err != nil {
		log.Fatalf("Failed to generate summary: %v", err)
	}

	keyPoints, err := goal.ChainOfThought(ctx, llmClient, fmt.Sprintf("Extract 3-5 key points from this analysis:\n%s", expertAnalysis))
	if err != nil {
		log.Fatalf("Failed to extract key points: %v", err)
	}

	fmt.Printf("Expert Topic: %s\n", expertTopic)
	fmt.Printf("Expert Analysis:\n%s\n\n", expertAnalysis)
	fmt.Printf("Summary (50 words):\n%s\n\n", summary)
	fmt.Printf("Key Points:\n%s\n", keyPoints)
}
