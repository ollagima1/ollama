package main

import (
	"strings"
)

func findStop(sequence string, stops []string) (bool, string) {
	for _, stop := range stops {
		if strings.Contains(sequence, stop) {
			return true, stop
		}
	}

	return false, ""
}

func containsStopSuffix(sequence string, stops []string) bool {
	for _, stop := range stops {
		for i := 1; i <= len(stop); i++ {
			if strings.HasSuffix(sequence, stop[:i]) {
				return true
			}
		}
	}

	return false
}

// truncateStop removes the provided stop string from pieces,
// returning the partial pieces with stop removed, including truncating
// the last piece if required (and signalling if this was the case)
func truncateStop(pieces []string, stop string) ([]string, bool) {
	joined := strings.Join(pieces, "")

	index := strings.Index(joined, stop)
	if index == -1 {
		return pieces, false
	}

	joined = joined[:index]

	// Split truncated string back into pieces of original lengths
	lengths := make([]int, len(pieces))
	for i, piece := range pieces {
		lengths[i] = len(piece)
	}

	var result []string
	tokenTruncated := false
	start := 0
	for _, length := range lengths {
		if start >= len(joined) {
			break
		}

		end := start + length
		if end > len(joined) {
			end = len(joined)
			tokenTruncated = true
		}
		result = append(result, joined[start:end])
		start = end
	}

	return result, tokenTruncated
}
