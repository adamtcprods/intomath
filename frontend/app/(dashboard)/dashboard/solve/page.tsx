import { Suspense } from "react";

import { SolveWorkspace } from "@/components/dashboard/solve-workspace";

export default function SolvePage() {
  return (
    <Suspense fallback={null}>
      <SolveWorkspace />
    </Suspense>
  );
}
