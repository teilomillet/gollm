package providers

type Provider interface {
	Name() string
	Endpoint() string
	Headers() map[string]string
	PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
	PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error)
	ParseResponse(body []byte) (string, error)
	SetExtraHeaders(extraHeaders map[string]string)
	HandleFunctionCalls(body []byte) ([]byte, error)
	SupportsJSONSchema() bool
}
