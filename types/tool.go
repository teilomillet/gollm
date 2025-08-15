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
