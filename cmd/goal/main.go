package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/teilomillet/goal"
)

func main() {
	configPath := flag.String("config", "", "Path to config file")
	promptType := flag.String("type", "raw", "Prompt type (raw, qa, cot, summarize)")
	verbose := flag.Bool("verbose", false, "Display verbose output including full prompt")
	showConfig := flag.Bool("show-config", false, "Display the loaded configuration")
	logLevel := flag.String("log-level", "", "Log level (debug, info, warn, error)")

	flag.Parse()

	var llmClient goal.LLM
	var err error

	if *logLevel != "" {
		llmClient, err = goal.NewLLM(*configPath, *logLevel)
	} else {
		llmClient, err = goal.NewLLM(*configPath)
	}
	handleError(err, true)

	if *showConfig {
		fmt.Printf("Loaded configuration:\n%+v\n\n", llmClient)
	}

	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawPrompt := strings.Join(flag.Args(), " ")

	ctx := context.Background()
	var response string

	switch *promptType {
	case "qa":
		response, err = goal.QuestionAnswer(ctx, llmClient, rawPrompt)
	case "cot":
		response, err = goal.ChainOfThought(ctx, llmClient, rawPrompt)
	case "summarize":
		response, err = goal.Summarize(ctx, llmClient, rawPrompt, 100) // Default to 100 words summary
	default:
		prompt := goal.NewPrompt(rawPrompt)
		response, _, err = llmClient.Generate(ctx, prompt.String())
	}

	handleError(err, true)

	if *verbose {
		fmt.Printf("Prompt Type: %s\nRaw Prompt: %s\n\nResponse:\n---------\n", *promptType, rawPrompt)
	}
	fmt.Println(response)
}

func handleError(err error, fatal bool) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		if fatal {
			os.Exit(1)
		}
	}
}
