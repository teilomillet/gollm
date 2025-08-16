// Package types contains common data types used across the gollm library.
// While the name is generic, it provides shared type definitions that prevent
// circular dependencies between packages.
package types

type Function struct {
	Parameters  map[string]any `json:"parameters"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
}

type Tool struct {
	Function Function `json:"function"`
	Type     string   `json:"type"`
}
