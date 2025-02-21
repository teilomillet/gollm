package data_extraction

import (
	"context"
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"
	"sync"

	"github.com/mauza/gollm"
	"github.com/mauza/gollm/presets"
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

// extractReview is a generic function to extract a review
func extractReview[T any](ctx context.Context, llm gollm.LLM, text string, withValidation bool) (*T, error) {
	var validationMsg string
	if withValidation {
		validationMsg = "with"
	} else {
		validationMsg = "without"
	}

	fmt.Printf("Extracting movie review %s validation...\n", validationMsg)
	review, err := presets.ExtractStructuredData[T](ctx, llm, text)
	if err != nil {
		return nil, fmt.Errorf("failed to extract movie review %s validation: %v", validationMsg, err)
	}
	return review, nil
}

func main() {
	fmt.Println("Starting application...")

	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		log.Fatalf("GROQ_API_KEY environment variable is not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama-3.1-70b-versatile"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxRetries(3),
		gollm.SetMaxTokens(2048),
		gollm.SetLogLevel(gollm.LogLevelWarn),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	text := `I recently watched "Inception" directed by Christopher Nolan. This mind-bending sci-fi thriller from 2010 
	         keeps you on the edge of your seat. With its intricate plot and stunning visuals, I'd rate it 9.5 out of 10. 
	         It seamlessly blends elements of science fiction, action, and psychological drama. The movie explores the concept 
	         of dream infiltration and leaves you questioning reality long after the credits roll.`

	ctx := context.Background()
	var wg sync.WaitGroup

	// Channels to collect results
	reviewChan := make(chan *MovieReview)
	reviewValidatedChan := make(chan *MovieReviewValidated)
	errorChan := make(chan error, 2) // Buffer for potential errors from both goroutines

	// Extract without validation
	wg.Add(1)
	go func() {
		defer wg.Done()
		review, err := extractReview[MovieReview](ctx, llm, text, false)
		if err != nil {
			errorChan <- err
			return
		}
		reviewChan <- review
	}()

	// Extract with validation
	wg.Add(1)
	go func() {
		defer wg.Done()
		reviewValidated, err := extractReview[MovieReviewValidated](ctx, llm, text, true)
		if err != nil {
			errorChan <- err
			return
		}
		reviewValidatedChan <- reviewValidated
	}()

	// Wait for both goroutines to complete
	go func() {
		wg.Wait()
		close(reviewChan)
		close(reviewValidatedChan)
		close(errorChan)
	}()

	// Collect and print results
	for {
		select {
		case review, ok := <-reviewChan:
			if !ok {
				reviewChan = nil
			} else {
				fmt.Printf("\n%s\n", strings.Repeat("=", 50))
				fmt.Println("\nExtracted Movie Review (without validation):")
				printReview(review)
				fmt.Printf("\n%s\n", strings.Repeat("=", 50))
			}
		case reviewValidated, ok := <-reviewValidatedChan:
			if !ok {
				reviewValidatedChan = nil
			} else {
				fmt.Printf("\n%s\n", strings.Repeat("=", 50))
				fmt.Println("\nExtracted Movie Review (with validation):")
				printReview(reviewValidated)
				fmt.Printf("\n%s\n", strings.Repeat("=", 50))
			}
		case err, ok := <-errorChan:
			if !ok {
				errorChan = nil
			} else {
				fmt.Printf("Error occurred: %v\n", err)
			}
		}

		if reviewChan == nil && reviewValidatedChan == nil && errorChan == nil {
			break
		}
	}

	fmt.Println("Application completed.")
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
