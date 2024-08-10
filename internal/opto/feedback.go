// File: internal/opto/feedback.go

package opto

import (
	"fmt"
	"strings"
)

// Feedback represents the output feedback from an LLM execution
type Feedback struct {
	Score    float64
	Message  string
	Metadata map[string]interface{}
}

// NewFeedback creates a new Feedback instance
func NewFeedback(score float64, message string) *Feedback {
	return &Feedback{
		Score:    score,
		Message:  message,
		Metadata: make(map[string]interface{}),
	}
}

// AddMetadata adds a key-value pair to the Feedback's metadata
func (f *Feedback) AddMetadata(key string, value interface{}) {
	f.Metadata[key] = value
}

// GetMetadata retrieves a value from the Feedback's metadata
func (f *Feedback) GetMetadata(key string) (interface{}, bool) {
	value, ok := f.Metadata[key]
	return value, ok
}

// IsBetterThan compares this Feedback to another and returns true if this one is better
func (f *Feedback) IsBetterThan(other *Feedback) bool {
	return f.Score > other.Score
}

// String returns a string representation of the Feedback
func (f *Feedback) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Score: %.4f\n", f.Score))
	sb.WriteString(fmt.Sprintf("Message: %s\n", f.Message))
	if len(f.Metadata) > 0 {
		sb.WriteString("Metadata:\n")
		for k, v := range f.Metadata {
			sb.WriteString(fmt.Sprintf("  %s: %v\n", k, v))
		}
	}
	return sb.String()
}

// AggregateFeedback combines multiple Feedback instances into one
func AggregateFeedback(feedbacks ...*Feedback) *Feedback {
	if len(feedbacks) == 0 {
		return NewFeedback(0, "No feedback available")
	}

	totalScore := 0.0
	var messages []string
	aggregatedMetadata := make(map[string]interface{})

	for _, f := range feedbacks {
		totalScore += f.Score
		messages = append(messages, f.Message)
		for k, v := range f.Metadata {
			aggregatedMetadata[k] = v
		}
	}

	avgScore := totalScore / float64(len(feedbacks))
	combinedMessage := strings.Join(messages, " | ")

	aggregated := NewFeedback(avgScore, combinedMessage)
	aggregated.Metadata = aggregatedMetadata

	return aggregated
}
