// File: cmd/goal/main.go

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/teilomillet/goal/llm"
)

func main() {
	configPath := flag.String("config", "", "Path to config file")
	promptType := flag.String("type", "raw", "Prompt type (raw, qa, cot, summarize)")
	verbose := flag.Bool("verbose", false, "Display verbose output including full prompt")
	showConfig := flag.Bool("show-config", false, "Display the loaded configuration")

	flag.Parse()

	config, err := llm.LoadConfig(*configPath)
	llm.HandleError(err, true)

	llm.InitLogging(config.LogLevel)

	if *showConfig {
		fmt.Printf("Loaded configuration:\n%+v\n\n", *config)
	}

	if len(flag.Args()) < 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <prompt>\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(1)
	}

	rawPrompt := strings.Join(flag.Args(), " ")
	prompt := llm.CreatePrompt(*promptType, rawPrompt)

	llmClient, err := llm.NewLLMFromConfig(config)
	llm.HandleError(err, true)

	response, fullPrompt, err := llmClient.Generate(context.Background(), prompt.String())
	llm.HandleError(err, true)

	if *verbose {
		fmt.Printf("Full Prompt:\n------------\n%s\n\nResponse:\n---------\n", fullPrompt)
	}
	fmt.Println(response)
}
