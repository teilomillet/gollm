package utils

import "time"

type DebugManager struct {
	logger  Logger
	options DebugOptions
}

type DebugOptions struct {
	LogPrompts     bool
	LogResponses   bool
	LogPerformance bool
}

func NewDebugManager(logger Logger, options DebugOptions) *DebugManager {
	return &DebugManager{
		logger:  logger,
		options: options,
	}
}

func (dm *DebugManager) LogPrompt(prompt string) {
	if dm.options.LogPrompts {
		dm.logger.Debug("Prompt", "content", prompt)
	}
}

func (dm *DebugManager) LogResponse(response string) {
	if dm.options.LogResponses {
		dm.logger.Debug("Response", "content", response)
	}
}

func (dm *DebugManager) LogPerformance(duration time.Duration) {
	if dm.options.LogPerformance {
		dm.logger.Debug("Performance", "duration", duration)
	}
}

func (dm *DebugManager) SetLogLevel(level LogLevel) {
	dm.logger.SetLevel(level)
}
