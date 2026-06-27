"use client";

import { useMutation } from "@tanstack/react-query";

import { apiClient } from "@/lib/api-client";
import type { SolveRequest, SolveResponse } from "@/features/solver/types";

export function useSolveProblem() {
  return useMutation({
    mutationFn: async (payload: SolveRequest) => {
      return apiClient<SolveResponse>("/solve", {
        method: "POST",
        body: JSON.stringify(payload),
      });
    },
  });
}
