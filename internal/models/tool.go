// Package models provides shared model definitions for the GoLLM library.
package models

// Function represents a function that can be called by the LLM.
type Function struct {
	Parameters  map[string]any `json:"parameters"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
}

// Tool represents a tool that the LLM can use.
type Tool struct {
	Function Function `json:"function"`
	Type     string   `json:"type"`
}
