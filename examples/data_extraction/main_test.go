package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/presets"
)

func containsSciFi(slice []string) bool {
	for _, s := range slice {
		if strings.EqualFold(s, "Science Fiction") {
			return true
		}
	}
	return false
}

func TestMovieReviewValidation(t *testing.T) {
	tests := []struct {
		name    string
		review  MovieReviewValidated
		wantErr bool
	}{
		{
			name: "valid review",
			review: MovieReviewValidated{
				Title:    "Inception",
				Director: "Christopher Nolan",
				Year:     2010,
				Rating:   9.5,
				Genres:   []string{"Science Fiction", "Action", "Thriller"},
				Summary:  "A mind-bending sci-fi thriller about dream infiltration that keeps you questioning reality.",
			},
			wantErr: false,
		},
		{
			name: "invalid title - empty",
			review: MovieReviewValidated{
				Title:    "",
				Director: "Christopher Nolan",
				Year:     2010,
				Rating:   9.5,
				Genres:   []string{"Science Fiction", "Action", "Thriller"},
				Summary:  "A mind-bending sci-fi thriller about dream infiltration that keeps you questioning reality.",
			},
			wantErr: true,
		},
		{
			name: "invalid year - too early",
			review: MovieReviewValidated{
				Title:    "Inception",
				Director: "Christopher Nolan",
				Year:     1800,
				Rating:   9.5,
				Genres:   []string{"Science Fiction", "Action", "Thriller"},
				Summary:  "A mind-bending sci-fi thriller about dream infiltration that keeps you questioning reality.",
			},
			wantErr: true,
		},
		{
			name: "invalid rating - too high",
			review: MovieReviewValidated{
				Title:    "Inception",
				Director: "Christopher Nolan",
				Year:     2010,
				Rating:   11.0,
				Genres:   []string{"Science Fiction", "Action", "Thriller"},
				Summary:  "A mind-bending sci-fi thriller about dream infiltration that keeps you questioning reality.",
			},
			wantErr: true,
		},
		{
			name: "invalid genres - too many",
			review: MovieReviewValidated{
				Title:    "Inception",
				Director: "Christopher Nolan",
				Year:     2010,
				Rating:   9.5,
				Genres:   []string{"Sci-Fi", "Action", "Thriller", "Drama", "Mystery", "Adventure"},
				Summary:  "A mind-bending sci-fi thriller about dream infiltration that keeps you questioning reality.",
			},
			wantErr: true,
		},
		{
			name: "invalid summary - too short",
			review: MovieReviewValidated{
				Title:    "Inception",
				Director: "Christopher Nolan",
				Year:     2010,
				Rating:   9.5,
				Genres:   []string{"Science Fiction", "Action", "Thriller"},
				Summary:  "Too short",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := gollm.Validate(&tt.review)
			if tt.wantErr {
				require.Error(t, err, "Validation should fail")
			} else {
				require.NoError(t, err, "Validation should pass")
			}
		})
	}
}

func TestExtractReview(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping extraction test in short mode")
	}

	// Check for API key
	apiKey := os.Getenv("GROQ_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: GROQ_API_KEY not set")
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("groq"),
		gollm.SetModel("llama-3.1-70b-versatile"),
		gollm.SetAPIKey(apiKey),
		gollm.SetMaxRetries(3),
		gollm.SetMaxTokens(2048),
		gollm.SetLogLevel(gollm.LogLevelWarn),
	)
	require.NoError(t, err, "Should create LLM instance")

	ctx := context.Background()

	// Test extraction without validation
	t.Run("extract_without_validation", func(t *testing.T) {
		text := `I recently watched "Inception" directed by Christopher Nolan. This mind-bending sci-fi thriller from 2010 
		         keeps you on the edge of your seat. With its intricate plot and stunning visuals, I'd rate it 9.5 out of 10. 
		         It seamlessly blends elements of science fiction, action, and psychological drama. The movie explores the concept 
		         of dream infiltration and leaves you questioning reality long after the credits roll.`

		extracted, err := presets.ExtractStructuredData[MovieReview](ctx, llm, text,
			gollm.WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output is a valid JSON object",
				"Do not include any backticks or formatting in the response",
				"Return only the JSON object",
			),
			gollm.WithOutput("JSON object only"),
		)
		require.NoError(t, err, "Should extract review without validation")
		assert.NotNil(t, extracted, "Review should not be nil")
		assert.Equal(t, "Inception", extracted.Title, "Should extract correct title")
		assert.Equal(t, "Christopher Nolan", extracted.Director, "Should extract correct director")
		assert.InEpsilon(t, 2010, extracted.Year, 0.001, "Should extract correct year")
		assert.InEpsilon(t, 9.5, extracted.Rating, 0.001, "Should extract correct rating")
		assert.True(t, containsSciFi(extracted.Genres), "Should extract genres")
		assert.NotEmpty(t, extracted.Summary, "Should extract summary")
	})

	// Test extraction with validation
	t.Run("extract_with_validation", func(t *testing.T) {
		text := `I recently watched "Inception" directed by Christopher Nolan. This mind-bending sci-fi thriller from 2010 
		         keeps you on the edge of your seat. With its intricate plot and stunning visuals, I'd rate it 9.5 out of 10. 
		         It seamlessly blends elements of science fiction, action, and psychological drama. The movie explores the concept 
		         of dream infiltration and leaves you questioning reality long after the credits roll.`

		extracted, err := presets.ExtractStructuredData[MovieReviewValidated](ctx, llm, text,
			gollm.WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output is a valid JSON object",
				"Do not include any backticks or formatting in the response",
				"Return only the JSON object",
			),
			gollm.WithOutput("JSON object only"),
		)
		require.NoError(t, err, "Should extract review with validation")
		assert.NotNil(t, extracted, "Review should not be nil")
		assert.Equal(t, "Inception", extracted.Title, "Should extract correct title")
		assert.Equal(t, "Christopher Nolan", extracted.Director, "Should extract correct director")
		assert.InEpsilon(t, 2010, extracted.Year, 0.001, "Should extract correct year")
		assert.InEpsilon(t, 9.5, extracted.Rating, 0.001, "Should extract correct rating")
		assert.True(t, containsSciFi(extracted.Genres), "Should extract genres")
		assert.NotEmpty(t, extracted.Summary, "Should extract summary")

		// Validate the extracted review
		err = gollm.Validate(extracted)
		require.NoError(t, err, "Extracted review should pass validation")
	})

	// Test concurrent extraction
	t.Run("concurrent_extraction", func(t *testing.T) {
		text := `I recently watched "Inception" directed by Christopher Nolan. This mind-bending sci-fi thriller from 2010 
		         keeps you on the edge of your seat. With its intricate plot and stunning visuals, I'd rate it 9.5 out of 10. 
		         It seamlessly blends elements of science fiction, action, and psychological drama. The movie explores the concept 
		         of dream infiltration and leaves you questioning reality long after the credits roll.`

		done := make(chan bool)
		review1 := MovieReview{}
		review2 := MovieReviewValidated{}
		var err1, err2 error
		var extracted1 *MovieReview
		var extracted2 *MovieReviewValidated

		// Extract both reviews concurrently
		go func() {
			extracted1, err1 = presets.ExtractStructuredData[MovieReview](ctx, llm, text,
				gollm.WithDirectives(
					"Extract all relevant information from the text",
					"Ensure the output is a valid JSON object",
					"Do not include any backticks or formatting in the response",
					"Return only the JSON object",
				),
				gollm.WithOutput("JSON object only"),
			)
			if err1 == nil && extracted1 != nil {
				review1 = *extracted1
			}
			done <- true
		}()

		go func() {
			extracted2, err2 = presets.ExtractStructuredData[MovieReviewValidated](ctx, llm, text,
				gollm.WithDirectives(
					"Extract all relevant information from the text",
					"Ensure the output is a valid JSON object",
					"Do not include any backticks or formatting in the response",
					"Return only the JSON object",
				),
				gollm.WithOutput("JSON object only"),
			)
			if err2 == nil && extracted2 != nil {
				review2 = *extracted2
			}
			done <- true
		}()

		// Wait for both extractions with timeout
		timeout := time.After(45 * time.Second)
		for range 2 {
			select {
			case <-done:
				// Extraction completed
			case <-timeout:
				t.Fatal("Concurrent extraction timed out")
			}
		}

		require.NoError(t, err1, "Should extract unvalidated review without error")
		require.NoError(t, err2, "Should extract validated review without error")
		assert.NotNil(t, extracted1, "Unvalidated review should not be nil")
		assert.NotNil(t, extracted2, "Validated review should not be nil")

		// Verify extracted values
		assert.Equal(t, "Inception", review1.Title, "Should extract correct title in concurrent test")
		assert.Equal(t, "Christopher Nolan", review1.Director, "Should extract correct director in concurrent test")
		assert.InEpsilon(t, 2010, review1.Year, 0.001, "Should extract correct year in concurrent test")
		assert.InEpsilon(t, 9.5, review1.Rating, 0.001, "Should extract correct rating in concurrent test")
		assert.True(
			t,
			containsSciFi(review1.Genres),
			"Should extract genres in concurrent test",
		)
		assert.NotEmpty(t, review1.Summary, "Should extract summary in concurrent test")

		assert.Equal(t, "Inception", review2.Title, "Should extract correct title in concurrent test")
		assert.Equal(t, "Christopher Nolan", review2.Director, "Should extract correct director in concurrent test")
		assert.InEpsilon(t, 2010, review2.Year, 0.001, "Should extract correct year in concurrent test")
		assert.InEpsilon(t, 9.5, review2.Rating, 0.001, "Should extract correct rating in concurrent test")
		assert.True(
			t,
			containsSciFi(review2.Genres),
			"Should extract genres in concurrent test",
		)
		assert.NotEmpty(t, review2.Summary, "Should extract summary in concurrent test")
	})

	// Test error handling
	t.Run("error_handling", func(t *testing.T) {
		// Test with empty text
		_, err := presets.ExtractStructuredData[MovieReview](ctx, llm, "",
			gollm.WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output is a valid JSON object",
				"Do not include any backticks or formatting in the response",
				"Return only the JSON object",
			),
			gollm.WithOutput("JSON object only"),
		)
		require.Error(t, err, "Should fail with empty text")

		// Test with invalid text
		_, err = presets.ExtractStructuredData[MovieReview](ctx, llm, "Not a movie review",
			gollm.WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output is a valid JSON object",
				"Do not include any backticks or formatting in the response",
				"Return only the JSON object",
			),
			gollm.WithOutput("JSON object only"),
		)
		require.Error(t, err, "Should fail with invalid text")

		// Test with context.TODO
		_, err = presets.ExtractStructuredData[MovieReview](context.TODO(), llm, "Some text",
			gollm.WithDirectives(
				"Extract all relevant information from the text",
				"Ensure the output is a valid JSON object",
				"Do not include any backticks or formatting in the response",
				"Return only the JSON object",
			),
			gollm.WithOutput("JSON object only"),
		)
		require.Error(t, err, "Should fail with nil context")
	})
}
