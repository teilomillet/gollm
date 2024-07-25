package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"reflect"

	"github.com/teilomillet/goal"
)

// MovieReview without validation tags
type MovieReview struct {
	Title    string   `json:"title"`
	Director string   `json:"director"`
	Year     int      `json:"year"`
	Rating   float64  `json:"rating"`
	Genres   []string `json:"genres"`
	Summary  string   `json:"summary"`
}

// MovieReviewValidated with corrected validation tags
type MovieReviewValidated struct {
	Title    string   `json:"title" validate:"required,min=1,max=200"`
	Director string   `json:"director" validate:"required,min=2,max=100"`
	Year     int      `json:"year" validate:"required,min=1888,max=2100"`
	Rating   float64  `json:"rating" validate:"required,min=0,max=10"`
	Genres   []string `json:"genres" validate:"required,min=1,max=5,dive,min=3,max=20"`
	Summary  string   `json:"summary" validate:"required,min=10,max=1000"`
}

// Where the magic happen
func main() {
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		log.Fatalf("GROQ_API_KEY environment variable is not set")
	}

	llm, err := goal.NewLLM(
		goal.SetProvider("groq"),
		goal.SetModel("llama-3.1-70b-versatile"),
		goal.SetAPIKey(apiKey),
		goal.SetMaxRetries(3),
		goal.SetMaxTokens(2048),
		goal.SetDebugLevel(goal.LogLevelDebug),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	text := `I recently watched "Inception" directed by Christopher Nolan. This mind-bending sci-fi thriller from 2010 
	         keeps you on the edge of your seat. With its intricate plot and stunning visuals, I'd rate it 9.5 out of 10. 
	         It seamlessly blends elements of science fiction, action, and psychological drama. The movie explores the concept 
	         of dream infiltration and leaves you questioning reality long after the credits roll.`

	// Example without validation
	fmt.Println("Extracting movie review without validation...")
	review, err := goal.ExtractStructuredData[MovieReview](context.Background(), llm, text)
	if err != nil {
		fmt.Printf("Failed to extract movie review without validation: %v\n", err)
	} else {
		fmt.Println("Extraction successful.")
		fmt.Println("\nExtracted Movie Review (without validation):")
		printReview(review)
	}

	// Example with validation
	fmt.Println("\nExtracting movie review with validation...")
	reviewValidated, err := goal.ExtractStructuredData[MovieReviewValidated](context.Background(), llm, text)
	if err != nil {
		fmt.Printf("Failed to extract movie review with validation: %v\n", err)
	} else {
		fmt.Println("Extraction successful.")
		fmt.Println("\nExtracted Movie Review (with validation):")
		printReview(reviewValidated)
	}
}

// printReview is a helper function to print the review in a more readable format
func printReview(review interface{}) {
	v := reflect.ValueOf(review).Elem()
	t := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := t.Field(i)
		value := v.Field(i)

		fmt.Printf("%-10s: ", field.Name)

		switch value.Kind() {
		case reflect.String:
			fmt.Printf("%s\n", value.String())
		case reflect.Int:
			fmt.Printf("%d\n", value.Int())
		case reflect.Float64:
			fmt.Printf("%.1f\n", value.Float())
		case reflect.Slice:
			fmt.Printf("%v\n", value.Interface())
		default:
			fmt.Printf("%v\n", value.Interface())
		}
	}
}

